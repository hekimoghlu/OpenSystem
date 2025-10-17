/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

#include <gbm.h>
#include <optional>
#include <wtf/Vector.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/unix/UnixFileDescriptor.h>
#include <xf86drmMode.h>

namespace WPE {

namespace DRM {

using Property = std::pair<uint32_t, uint64_t>;

class Crtc {
    WTF_MAKE_TZONE_ALLOCATED(Crtc);
public:
    struct Properties {
        Property active { 0, 0 };
        Property modeID { 0, 0 };
    };

    static std::unique_ptr<Crtc> create(int, drmModeCrtc*, unsigned);
    Crtc(drmModeCrtc*, unsigned, Properties&&);
    ~Crtc() = default;

    uint32_t id() const { return m_id; }
    unsigned index() const { return m_index; }
    uint32_t x() const { return m_x; }
    uint32_t y() const { return m_y; }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    const std::optional<drmModeModeInfo>& currentMode() const { return m_currentMode; }
    const Properties& properties() const { return m_properties; }

    bool modeIsCurrent(drmModeModeInfo*) const;

private:
    uint32_t m_id { 0 };
    unsigned m_index { 0 };
    uint32_t m_x { 0 };
    uint32_t m_y { 0 };
    uint32_t m_width { 0 };
    uint32_t m_height { 0 };
    std::optional<drmModeModeInfo> m_currentMode;
    Properties m_properties;
};

class Connector {
    WTF_MAKE_TZONE_ALLOCATED(Connector);
public:
    struct Properties {
        Property crtcID { 0, 0 };
        Property linkStatus { 0, 0 };
        Property DPMS { 0, 0 };
    };

    static std::unique_ptr<Connector> create(int, drmModeConnector*);
    Connector(drmModeConnector*, Properties&&);
    ~Connector() = default;

    uint32_t id() const { return m_id; }
    uint32_t encoderID() const { return m_encoderID; }
    uint32_t widthMM() const { return m_widthMM; }
    uint32_t heightMM() const { return m_heightMM; }
    const Vector<drmModeModeInfo>& modes() const { return m_modes; }
    const std::optional<unsigned> preferredModeIndex() const { return m_preferredModeIndex; }
    const Properties& properties() const { return m_properties; }

private:
    uint32_t m_id { 0 };
    uint32_t m_encoderID { 0 };
    uint32_t m_widthMM { 0 };
    uint32_t m_heightMM { 0 };
    Vector<drmModeModeInfo> m_modes;
    std::optional<unsigned> m_preferredModeIndex;
    Properties m_properties;
};

class Plane {
    WTF_MAKE_TZONE_ALLOCATED(Plane);
public:
    enum class Type : uint8_t {
        Primary = DRM_PLANE_TYPE_PRIMARY,
        Cursor = DRM_PLANE_TYPE_CURSOR
    };

    struct Properties {
        Property crtcID { 0, 0 };
        Property crtcX { 0, 0 };
        Property crtcY { 0, 0 };
        Property crtcW { 0, 0 };
        Property crtcH { 0, 0 };
        Property fbID { 0, 0 };
        Property srcX { 0, 0 };
        Property srcY { 0, 0 };
        Property srcW { 0, 0 };
        Property srcH { 0, 0 };
        Property fbDamageClips { 0, 0 };
        Property inFenceFD { 0, 0 };
    };

    struct Format {
        uint32_t format { 0 };
        Vector<uint64_t> modifiers;
    };

    static std::unique_ptr<Plane> create(int, Type, drmModePlane*, bool);
    Plane(drmModePlane*, Vector<Format>&&, Properties&&);
    ~Plane() = default;

    uint32_t id() const { return m_id; }
    const Properties& properties() const { return m_properties; }
    const Vector<Format>& formats() const { return m_formats; }

    bool canBeUsedWithCrtc(const Crtc& crtc) const { return !!(m_possibleCrtcs & (1 << crtc.index())); }
    bool supportsFormat(uint32_t, uint64_t) const;

private:
    uint32_t m_id { 0 };
    uint32_t m_possibleCrtcs { 0 };
    Vector<Format> m_formats;
    Properties m_properties;
};

class Buffer {
    WTF_MAKE_TZONE_ALLOCATED(Buffer);
public:
    static std::unique_ptr<Buffer> create(struct gbm_bo*);
    Buffer(struct gbm_bo*, uint32_t);
    ~Buffer();

    struct gbm_bo* bufferObject() const { return m_bufferObject; }
    uint32_t frameBufferID() const { return m_frameBufferID; }
    void setFenceFD(WTF::UnixFileDescriptor&& fenceFD) { m_fenceFD = WTFMove(fenceFD); }
    const WTF::UnixFileDescriptor& fenceFD() const { return m_fenceFD; }

private:
    struct gbm_bo* m_bufferObject { nullptr };
    uint32_t m_frameBufferID { 0 };
    mutable WTF::UnixFileDescriptor m_fenceFD;
};

} // namespace DRM

} // namespace WPE
