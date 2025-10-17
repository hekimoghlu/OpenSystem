/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

// This include order is necessary to enforce the GBM EGL platform.
#include <gbm.h>
#include <epoxy/egl.h>

#include <QHoverEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QPointer>
#include <QWheelEvent>
#include <wpe/fdo-egl.h>
#include <wpe/fdo.h>

class WPEQtView;

class Q_DECL_EXPORT WPEQtViewBackend {
public:
    static std::unique_ptr<WPEQtViewBackend> create(const QSizeF&, QPointer<QOpenGLContext>, EGLDisplay, QPointer<WPEQtView>);
    WPEQtViewBackend(const QSizeF&, EGLDisplay, EGLContext, QPointer<QOpenGLContext>, QPointer<WPEQtView>);
    virtual ~WPEQtViewBackend();

    void resize(const QSizeF&);
    GLuint texture(QOpenGLContext*);
    bool hasValidSurface() const { return m_surface.isValid(); };

    void dispatchHoverEnterEvent(QHoverEvent*);
    void dispatchHoverLeaveEvent(QHoverEvent*);
    void dispatchHoverMoveEvent(QHoverEvent*);

    void dispatchMousePressEvent(QMouseEvent*);
    void dispatchMouseReleaseEvent(QMouseEvent*);
    void dispatchWheelEvent(QWheelEvent*);

    void dispatchKeyEvent(QKeyEvent*, bool state);

    void dispatchTouchEvent(QTouchEvent*);

    struct wpe_view_backend* backend() const { return wpe_view_backend_exportable_fdo_get_view_backend(m_exportable); };

private:
    void displayImage(struct wpe_fdo_egl_exported_image*);
    uint32_t modifiers() const;

    EGLDisplay m_eglDisplay { nullptr };
    EGLContext m_eglContext { nullptr };
    struct wpe_view_backend_exportable_fdo* m_exportable { nullptr };
    struct wpe_fdo_egl_exported_image* m_lockedImage { nullptr };

    QPointer<WPEQtView> m_view;
    QOffscreenSurface m_surface;
    QSizeF m_size;
    GLuint m_textureId { 0 };
    unsigned m_program { 0 };
    unsigned m_textureUniform { 0 };

    bool m_hovering { false };
    uint32_t m_mouseModifiers { 0 };
    uint32_t m_keyboardModifiers { 0 };
    uint32_t m_mousePressedButton { 0 };
};
