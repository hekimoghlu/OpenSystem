/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include "config.h"
#include "WPEScreenDRM.h"

#include "WPEScreenDRMPrivate.h"
#include <wtf/glib/WTFGType.h>

/**
 * WPEScreenDRM:
 *
 */
struct _WPEScreenDRMPrivate {
    std::unique_ptr<WPE::DRM::Crtc> crtc;
    drmModeModeInfo mode;
};
WEBKIT_DEFINE_FINAL_TYPE(WPEScreenDRM, wpe_screen_drm, WPE_TYPE_SCREEN, WPEScreen)

static void wpeScreenDRMInvalidate(WPEScreen* screen)
{
    auto* priv = WPE_SCREEN_DRM(screen)->priv;
    priv->crtc = nullptr;
}

static void wpeScreenDRMDispose(GObject* object)
{
    wpeScreenDRMInvalidate(WPE_SCREEN(object));

    G_OBJECT_CLASS(wpe_screen_drm_parent_class)->dispose(object);
}

static void wpe_screen_drm_class_init(WPEScreenDRMClass* screenDRMClass)
{
    GObjectClass* objectClass = G_OBJECT_CLASS(screenDRMClass);
    objectClass->dispose = wpeScreenDRMDispose;

    WPEScreenClass* screenClass = WPE_SCREEN_CLASS(screenDRMClass);
    screenClass->invalidate = wpeScreenDRMInvalidate;
}

WPEScreen* wpeScreenDRMCreate(std::unique_ptr<WPE::DRM::Crtc>&& crtc, const WPE::DRM::Connector& connector)
{
    auto* screen = WPE_SCREEN(g_object_new(WPE_TYPE_SCREEN_DRM, "id", crtc->id(), nullptr));
    auto* priv = WPE_SCREEN_DRM(screen)->priv;
    priv->crtc = WTFMove(crtc);

    wpe_screen_set_physical_size(screen, connector.widthMM(), connector.heightMM());

    if (const auto& mode = priv->crtc->currentMode())
        priv->mode = mode.value();
    else {
        if (const auto& preferredModeIndex = connector.preferredModeIndex())
            priv->mode = connector.modes()[preferredModeIndex.value()];
        else {
            int area = 0;
            for (const auto& mode : connector.modes()) {
                int modeArea = mode.hdisplay * mode.vdisplay;
                if (modeArea > area) {
                    priv->mode = mode;
                    area = modeArea;
                }
            }
        }
    }

    uint32_t refresh = [](drmModeModeInfo* info) -> uint32_t {
        uint64_t refresh = (info->clock * 1000000LL / info->htotal + info->vtotal / 2) / info->vtotal;
        if (info->flags & DRM_MODE_FLAG_INTERLACE)
            refresh *= 2;
        if (info->flags & DRM_MODE_FLAG_DBLSCAN)
            refresh /= 2;
        if (info->vscan > 1)
            refresh /= info->vscan;

        return refresh;
    }(&priv->mode);

    wpe_screen_set_refresh_rate(screen, refresh);

    return WPE_SCREEN(screen);
}

drmModeModeInfo* wpeScreenDRMGetMode(WPEScreenDRM* screen)
{
    return &screen->priv->mode;
}

const WPE::DRM::Crtc wpeScreenDRMGetCrtc(WPEScreenDRM* screen)
{
    return *screen->priv->crtc;
}

/**
 * wpe_screen_drm_get_crtc_index: (skip)
 * @screen: a #WPEScreenDRM
 *
 * Get the DRM CRTC index of @screen
 *
 * Returns: the CRTC index
 */
guint wpe_screen_drm_get_crtc_index(WPEScreenDRM* screen)
{
    g_return_val_if_fail(WPE_IS_SCREEN_DRM(screen), 0);

    return screen->priv->crtc->index();
}
