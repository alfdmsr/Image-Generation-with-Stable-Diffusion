"""StudioAI Streamlit Web UI — deployed on Modal (CPU UI + Serverless GPU backend)."""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas

import logic

# Page Config
st.set_page_config(page_title="StudioAI", layout="wide", page_icon="🎨")

st.title("🎨 StudioAI: Creating Amazing Art with Stable Diffusion")
st.caption("Powered by Stable Diffusion v1.5 · UI on CPU, Inference on Serverless GPU (Modal)")

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parameters")

    steps = st.slider("Quality Steps", 15, 50, 30)
    cfg = st.slider("Creativity (CFG)", 1.0, 20.0, 7.5, step=0.5)
    seed = st.number_input("Seed Control", min_value=0, value=42, step=1)

    st.divider()

    st.subheader("🚀 Advanced")
    scheduler_name = st.selectbox("Scheduler", ["Euler A", "DPM++", "DDIM"])
    num_images = st.slider("Batch Size (Jumlah Gambar)", 1, 4, 1)

    st.divider()
    if st.button("🧹 Flush GPU VRAM"):
        with st.spinner("Membersihkan cache VRAM GPU..."):
            try:
                msg = logic.remote_flush_memory()
                st.toast(f"✅ {msg}")
            except Exception as exc:
                st.error(f"Gagal flush VRAM: {exc}")

    st.info(
        "💡 **Info Cold Start:** Permintaan pertama setelah idle ±2 menit "
        "membutuhkan ~20–40 detik untuk menyalakan GPU. Generasi berikutnya jauh lebih cepat."
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_gen, tab_edit = st.tabs(["✨ GENERATE", "🛠️ EDIT"])

# ===========================================================================
# TAB 1: TEXT-TO-IMAGE GENERATION
# ===========================================================================
with tab_gen:
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.subheader("Input Blueprint")
        with st.form(key="gen_form"):
            prompt = st.text_area(
                "Prompt",
                "a cute robot in a futuristic city, 8k, masterpiece",
                height=150,
            )
            neg_prompt = st.text_input(
                "Negative Prompt",
                "blurry, bad anatomy, worst quality",
            )
            submit_gen = st.form_submit_button("🚀 Initialize Generation", type="primary")

        if submit_gen:
            if not prompt.strip():
                st.warning("⚠️ Prompt tidak boleh kosong.")
            else:
                with st.spinner("Processing Image on Modal GPU..."):
                    try:
                        generated_list = logic.remote_generate(
                            prompt=prompt,
                            neg_prompt=neg_prompt,
                            seed=int(seed),
                            steps=int(steps),
                            cfg=float(cfg),
                            num_images=int(num_images),
                            scheduler_name=scheduler_name,
                        )
                        st.session_state["generated_images"] = generated_list
                        st.session_state["last_prompt"] = prompt
                        st.session_state["last_neg_prompt"] = neg_prompt
                        if generated_list:
                            st.session_state["current_image"] = generated_list[0]
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Gagal generate gambar: {exc}")

    with c2:
        st.subheader("Visual Output")

        if "generated_images" in st.session_state:
            imgs = st.session_state["generated_images"]

            if len(imgs) > 1:
                cols = st.columns(2)
                for idx, img in enumerate(imgs):
                    with cols[idx % 2]:
                        st.image(img, caption=f"Img {idx + 1}", use_container_width=True)
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button(f"Edit Img {idx + 1}", key=f"sel_{idx}"):
                                st.session_state["current_image"] = img
                                st.toast(f"✅ Image {idx + 1} Selected for Editing!")
                        with b2:
                            st.download_button(
                                label="⬇️ PNG",
                                data=logic.pil_to_bytes(img),
                                file_name=f"studioai_gen_{idx + 1}.png",
                                mime="image/png",
                                key=f"dl_{idx}",
                            )
            elif len(imgs) == 1:
                st.image(imgs[0], caption="Result", use_container_width=True)
                st.download_button(
                    label="⬇️ Download PNG",
                    data=logic.pil_to_bytes(imgs[0]),
                    file_name="studioai_result.png",
                    mime="image/png",
                )
        else:
            st.info("👈 Masukkan prompt di panel kiri dan tekan **Initialize Generation**.")

# ===========================================================================
# TAB 2: EDIT (INPAINTING & OUTPAINTING)
# ===========================================================================
with tab_edit:
    with st.expander("📤 Atau Upload Gambar Sendiri untuk Diedit", expanded="current_image" not in st.session_state):
        uploaded = st.file_uploader("Upload gambar (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            uploaded_id = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("last_uploaded_id") != uploaded_id:
                up_img = Image.open(uploaded).convert("RGB")
                fit_w, fit_h = logic.fit_dimensions(up_img.width, up_img.height, max_side=512, min_side=512)
                st.session_state["current_image"] = up_img.resize((fit_w, fit_h), Image.LANCZOS)
                st.session_state["last_uploaded_id"] = uploaded_id
                st.session_state["canvas_key"] = int(st.session_state.get("canvas_key", 0)) + 1
                st.rerun()

    if "current_image" in st.session_state:
        source_img = st.session_state["current_image"]

        if isinstance(source_img, list):
            source_img = source_img[0]
            st.session_state["current_image"] = source_img

        W, H = source_img.size

        mode = st.radio(
            "Select Mode:",
            ["Inpainting (Edit Objek)", "Outpainting (Zoom Out)"],
            horizontal=True,
        )
        st.divider()

        # -------------------------------------------------------------------
        # Mode 1: Inpainting
        # -------------------------------------------------------------------
        if mode == "Inpainting (Edit Objek)":
            col_tools, col_result = st.columns([1, 1], gap="large")

            if "canvas_key" not in st.session_state:
                st.session_state["canvas_key"] = 0

            # Keep canvas display reasonably sized while preserving aspect ratio
            display_max = 512
            if max(W, H) > display_max:
                scale = display_max / max(W, H)
                canvas_w, canvas_h = int(W * scale), int(H * scale)
                canvas_bg = source_img.resize((canvas_w, canvas_h), Image.LANCZOS)
            else:
                canvas_w, canvas_h = W, H
                canvas_bg = source_img

            dynamic_key = f"canvas_{st.session_state['canvas_key']}_{id(source_img)}"

            with col_tools:
                st.subheader("✍️ Draw Mask")
                brush_width = st.slider("Ukuran Kuas", 5, 60, 20)
                st.caption("Warnai area yang ingin diubah dengan kuas putih.")

                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1.0)",
                    stroke_width=brush_width,
                    stroke_color="#FFFFFF",
                    background_image=canvas_bg,
                    update_streamlit=True,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode="freedraw",
                    key=dynamic_key,
                )

            with col_result:
                st.subheader("Settings")
                with st.form("inpaint_input"):
                    edit_prompt = st.text_input(
                        "Prompt Baru (Ganti jadi apa?)",
                        "a pair of sunglasses",
                    )
                    edit_neg_prompt = st.text_input(
                        "Negative Prompt (Opsional)",
                        st.session_state.get("last_neg_prompt", "blurry, low quality"),
                    )
                    strength = st.slider(
                        "Strength (Seberapa kuat perubahannya?)",
                        0.1,
                        1.0,
                        0.85,
                        step=0.05,
                    )
                    submit_inpaint = st.form_submit_button("⚡ Execute Inpainting", type="primary")

                st.download_button(
                    label="⬇️ Download Gambar Saat Ini",
                    data=logic.pil_to_bytes(source_img),
                    file_name="studioai_edited.png",
                    mime="image/png",
                )

                if submit_inpaint:
                    if canvas_result.image_data is not None and np.max(canvas_result.image_data[:, :, 3]) > 0:
                        with st.spinner("Processing Inpainting on Modal GPU..."):
                            mask_data = canvas_result.image_data[:, :, 3].copy()
                            mask_data[mask_data > 0] = 255
                            mask_image = Image.fromarray(mask_data.astype("uint8"), mode="L")

                            if mask_image.size != source_img.size:
                                mask_image = mask_image.resize(source_img.size, resample=Image.NEAREST)

                            # Dilate mask slightly for smoother coverage
                            mask_image = mask_image.filter(ImageFilter.MaxFilter(15))

                            with st.expander("🕵️ Lihat Masker Final (Debug)"):
                                st.image(mask_image, caption="Masker Tajam (Tanpa Blur)", width=200)

                            try:
                                result_img = logic.remote_inpaint(
                                    image=source_img,
                                    mask=mask_image,
                                    prompt=edit_prompt,
                                    strength=float(strength),
                                    neg_prompt=edit_neg_prompt,
                                    seed=int(seed),
                                    steps=int(steps),
                                    cfg=float(cfg),
                                    scheduler_name=scheduler_name,
                                )
                                st.session_state["current_image"] = result_img
                                st.session_state["canvas_key"] = int(st.session_state.get("canvas_key", 0)) + 1
                                st.success("Inpainting Selesai!")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Error pada proses inpainting: {exc}")
                    else:
                        st.warning("⚠️ Silakan coret gambar terlebih dahulu!")

        # -------------------------------------------------------------------
        # Mode 2: Outpainting (Zoom Out)
        # -------------------------------------------------------------------
        elif mode == "Outpainting (Zoom Out)":
            c_out_1, c_out_2 = st.columns([1, 1], gap="large")

            with c_out_1:
                st.subheader("Original")
                st.image(source_img, caption=f"Gambar saat ini ({W}×{H})", use_container_width=True)
                st.download_button(
                    label="⬇️ Download Gambar Saat Ini",
                    data=logic.pil_to_bytes(source_img),
                    file_name="studioai_outpainted.png",
                    mime="image/png",
                )

            with c_out_2:
                st.subheader("Expansion")
                default_out_prompt = (
                    f"wide angle view of {st.session_state.get('last_prompt', 'the scene')}, "
                    "detailed background, 8k"
                )
                with st.form("outpaint_input"):
                    expand_px = st.slider("Expand Pixels (per sisi)", 64, 192, 128, step=32)
                    out_prompt = st.text_input(
                        "Prompt Deskriptif (Jelaskan gambar UTUH)",
                        default_out_prompt,
                    )
                    out_neg_prompt = st.text_input(
                        "Negative Prompt (Opsional)",
                        st.session_state.get("last_neg_prompt", "blurry, border, frame, low quality"),
                    )
                    submit_outpaint = st.form_submit_button("🔍 Zoom Out (Expand)", type="primary")

                if submit_outpaint:
                    with st.spinner("Expanding Canvas on Modal GPU..."):
                        try:
                            canvas_ready, mask_ready = logic.prepare_outpainting(
                                source_img, expand_pixels=int(expand_px)
                            )
                            final_result = logic.remote_inpaint(
                                image=canvas_ready,
                                mask=mask_ready,
                                prompt=out_prompt,
                                strength=1.0,
                                neg_prompt=out_neg_prompt,
                                seed=int(seed),
                                steps=int(steps),
                                cfg=float(cfg),
                                scheduler_name=scheduler_name,
                            )
                            st.session_state["current_image"] = final_result
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error pada proses Outpainting: {exc}")

    else:
        st.info("👈 Belum ada gambar. Silakan ke Tab **✨ GENERATE** atau upload gambar di atas.")
