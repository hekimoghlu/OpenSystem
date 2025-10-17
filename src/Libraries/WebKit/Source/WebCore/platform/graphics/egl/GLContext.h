/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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

#include "GLContextWrapper.h"
#include "IntSize.h"
#include "PlatformDisplay.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

#if !PLATFORM(GTK) && !PLATFORM(WPE)
#include <EGL/eglplatform.h>
typedef EGLNativeWindowType GLNativeWindowType;
#else
typedef uint64_t GLNativeWindowType;
#endif

#if USE(WPE_RENDERER)
struct wpe_renderer_backend_egl_offscreen_target;
#endif

typedef void* GCGLContext;
typedef void* EGLConfig;
typedef void* EGLContext;
typedef void* EGLDisplay;
typedef void* EGLSurface;

namespace WebCore {

class GLContext final : public GLContextWrapper {
    WTF_MAKE_TZONE_ALLOCATED(GLContext);
    WTF_MAKE_NONCOPYABLE(GLContext);
public:
    WEBCORE_EXPORT static std::unique_ptr<GLContext> create(GLNativeWindowType, PlatformDisplay&);
    static std::unique_ptr<GLContext> createOffscreen(PlatformDisplay&);
    static std::unique_ptr<GLContext> createSharing(PlatformDisplay&);

    static GLContext* current();
    static bool isExtensionSupported(const char* extensionList, const char* extension);
    static unsigned versionFromString(const char* versionString);

    static const char* errorString(int statusCode);
    static const char* lastErrorString();

    enum EGLSurfaceType { PbufferSurface, WindowSurface, PixmapSurface, Surfaceless };
    GLContext(PlatformDisplay&, EGLContext, EGLSurface, EGLConfig, EGLSurfaceType);
#if USE(WPE_RENDERER)
    GLContext(PlatformDisplay&, EGLContext, EGLSurface, EGLConfig, struct wpe_renderer_backend_egl_offscreen_target*);
#endif
    WEBCORE_EXPORT ~GLContext();

    PlatformDisplay& display() const { return m_display; }
    unsigned version();
    EGLConfig config() const { return m_config; }

    WEBCORE_EXPORT bool makeContextCurrent();
    bool unmakeContextCurrent();
    WEBCORE_EXPORT void swapBuffers();
    GCGLContext platformContext() const;

    struct GLExtensions {
        bool OES_texture_npot { false };
        bool EXT_unpack_subimage { false };
        bool APPLE_sync { false };
        bool OES_packed_depth_stencil { false };
    };
    const GLExtensions& glExtensions() const;

    class ScopedGLContext {
        WTF_MAKE_NONCOPYABLE(ScopedGLContext);
    public:
        explicit ScopedGLContext(std::unique_ptr<GLContext>&&);
        ~ScopedGLContext();
    private:
        struct {
            GLContext* glContext { nullptr };
            EGLDisplay display { nullptr };
            EGLContext context { nullptr };
            EGLSurface readSurface { nullptr };
            EGLSurface drawSurface { nullptr };
        } m_previous;
        std::unique_ptr<GLContext> m_context;
    };

    class ScopedGLContextCurrent {
        WTF_MAKE_NONCOPYABLE(ScopedGLContextCurrent);
    public:
        explicit ScopedGLContextCurrent(GLContext&);
        ~ScopedGLContextCurrent();
    private:
        struct {
            GLContext* glContext { nullptr };
            EGLDisplay display { nullptr };
            EGLContext context { nullptr };
            EGLSurface readSurface { nullptr };
            EGLSurface drawSurface { nullptr };
        } m_previous;
        GLContext& m_context;
    };

private:
    static EGLContext createContextForEGLVersion(PlatformDisplay&, EGLConfig, EGLContext);

    static std::unique_ptr<GLContext> createWindowContext(GLNativeWindowType, PlatformDisplay&, EGLContext sharingContext = nullptr);
    static std::unique_ptr<GLContext> createPbufferContext(PlatformDisplay&, EGLContext sharingContext = nullptr);
    static std::unique_ptr<GLContext> createSurfacelessContext(PlatformDisplay&, EGLContext sharingContext = nullptr);
#if USE(WPE_RENDERER)
    static std::unique_ptr<GLContext> createWPEContext(PlatformDisplay&, EGLContext sharingContext = nullptr);
    static EGLSurface createWindowSurfaceWPE(EGLDisplay, EGLConfig, GLNativeWindowType);
    void destroyWPETarget();
#endif

    static bool getEGLConfig(PlatformDisplay&, EGLConfig*, EGLSurfaceType);

    // GLContextWrapper
    GLContextWrapper::Type type() const override { return GLContextWrapper::Type::Native; }
    bool makeCurrentImpl() override;
    bool unmakeCurrentImpl() override;

    PlatformDisplay& m_display;
    unsigned m_version { 0 };
    EGLContext m_context { nullptr };
    EGLSurface m_surface { nullptr };
    EGLConfig m_config { nullptr };
    EGLSurfaceType m_type;
#if USE(WPE_RENDERER)
    struct wpe_renderer_backend_egl_offscreen_target* m_wpeTarget { nullptr };
#endif
    mutable GLExtensions m_glExtensions;
};

} // namespace WebCore
