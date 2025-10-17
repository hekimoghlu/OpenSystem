/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class DeprecatedGlobalSettings {
public:
#if USE(AVFOUNDATION)
    WEBCORE_EXPORT static void setAVFoundationEnabled(bool);
    static bool isAVFoundationEnabled() { return shared().m_AVFoundationEnabled; }
#endif

#if USE(GSTREAMER)
    WEBCORE_EXPORT static void setGStreamerEnabled(bool);
    static bool isGStreamerEnabled() { return shared().m_GStreamerEnabled; }
#endif

    WEBCORE_EXPORT static void setMockScrollbarsEnabled(bool);
    static bool mockScrollbarsEnabled() { return shared().m_mockScrollbarsEnabled; }

    WEBCORE_EXPORT static void setUsesOverlayScrollbars(bool);
    static bool usesOverlayScrollbars() { return shared().m_usesOverlayScrollbars; }

    static bool lowPowerVideoAudioBufferSizeEnabled() { return shared().m_lowPowerVideoAudioBufferSizeEnabled; }
    static void setLowPowerVideoAudioBufferSizeEnabled(bool flag) { shared().m_lowPowerVideoAudioBufferSizeEnabled = flag; }

    static bool trackingPreventionEnabled() { return shared().m_trackingPreventionEnabled; }
    WEBCORE_EXPORT static void setTrackingPreventionEnabled(bool);

#if PLATFORM(IOS_FAMILY)
    WEBCORE_EXPORT static void setAudioSessionCategoryOverride(unsigned);
    static unsigned audioSessionCategoryOverride();

    WEBCORE_EXPORT static void setNetworkInterfaceName(const String&);
    static const String& networkInterfaceName() { return shared().m_networkInterfaceName; }

    static void setDisableScreenSizeOverride(bool flag) { shared().m_disableScreenSizeOverride = flag; }
    static bool disableScreenSizeOverride() { return shared().m_disableScreenSizeOverride; }

    static void setShouldOptOutOfNetworkStateObservation(bool flag) { shared().m_shouldOptOutOfNetworkStateObservation = flag; }
    static bool shouldOptOutOfNetworkStateObservation() { return shared().m_shouldOptOutOfNetworkStateObservation; }
#endif

#if USE(AUDIO_SESSION)
    WEBCORE_EXPORT static void setShouldManageAudioSessionCategory(bool);
    WEBCORE_EXPORT static bool shouldManageAudioSessionCategory();
#endif

    WEBCORE_EXPORT static void setAllowsAnySSLCertificate(bool);
    WEBCORE_EXPORT static bool allowsAnySSLCertificate();

    static void setCustomPasteboardDataEnabled(bool isEnabled) { shared().m_isCustomPasteboardDataEnabled = isEnabled; }
    static bool customPasteboardDataEnabled() { return shared().m_isCustomPasteboardDataEnabled; }

    static void setAttrStyleEnabled(bool isEnabled) { shared().m_attrStyleEnabled = isEnabled; }
    static bool attrStyleEnabled() { return shared().m_attrStyleEnabled; }

    static void setWebSQLEnabled(bool isEnabled) { shared().m_webSQLEnabled = isEnabled; }
    static bool webSQLEnabled() { return shared().m_webSQLEnabled; }

#if ENABLE(ATTACHMENT_ELEMENT)
    static void setAttachmentElementEnabled(bool areEnabled) { shared().m_isAttachmentElementEnabled = areEnabled; }
    static bool attachmentElementEnabled() { return shared().m_isAttachmentElementEnabled; }
#endif

    static bool webRTCAudioLatencyAdaptationEnabled() { return shared().m_isWebRTCAudioLatencyAdaptationEnabled; }
    static void setWebRTCAudioLatencyAdaptationEnabled(bool isEnabled) { shared().m_isWebRTCAudioLatencyAdaptationEnabled = isEnabled; }

    static void setReadableByteStreamAPIEnabled(bool isEnabled) { shared().m_isReadableByteStreamAPIEnabled = isEnabled; }
    static bool readableByteStreamAPIEnabled() { return shared().m_isReadableByteStreamAPIEnabled; }

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    static void setIsAccessibilityIsolatedTreeEnabled(bool isEnabled) { shared().m_accessibilityIsolatedTree = isEnabled; }
    static bool isAccessibilityIsolatedTreeEnabled() { return shared().m_accessibilityIsolatedTree; }
#endif

#if ENABLE(AX_THREAD_TEXT_APIS)
    static void setAccessibilityThreadTextApisEnabled(bool isEnabled) { shared().m_accessibilityThreadTextApis = isEnabled; }
    static bool accessibilityThreadTextApisEnabled() { return shared().m_accessibilityThreadTextApis; }
#endif

    static void setArePDFImagesEnabled(bool isEnabled) { shared().m_arePDFImagesEnabled = isEnabled; }
    static bool arePDFImagesEnabled() { return shared().m_arePDFImagesEnabled; }

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    static void setBuiltInNotificationsEnabled(bool isEnabled) { shared().m_builtInNotificationsEnabled = isEnabled; }
    WEBCORE_EXPORT static bool builtInNotificationsEnabled();
#endif

#if ENABLE(MODEL_ELEMENT)
    static void setModelDocumentEnabled(bool isEnabled) { shared().m_modelDocumentEnabled = isEnabled; }
    static bool modelDocumentEnabled() { return shared().m_modelDocumentEnabled; }
#endif


private:
    WEBCORE_EXPORT static DeprecatedGlobalSettings& shared();
    DeprecatedGlobalSettings() = default;
    ~DeprecatedGlobalSettings() = default;

#if USE(AVFOUNDATION)
    bool m_AVFoundationEnabled { true };
#endif

#if USE(GSTREAMER)
    bool m_GStreamerEnabled { true };
#endif

    bool m_mockScrollbarsEnabled { false };
    bool m_usesOverlayScrollbars { false };

#if PLATFORM(IOS_FAMILY)
    String m_networkInterfaceName;
    bool m_shouldOptOutOfNetworkStateObservation { false };
    bool m_disableScreenSizeOverride { false };
#endif

    bool m_lowPowerVideoAudioBufferSizeEnabled { false };
    bool m_trackingPreventionEnabled { false };
    bool m_allowsAnySSLCertificate { false };

    bool m_isCustomPasteboardDataEnabled { false };
    bool m_attrStyleEnabled { false };
    bool m_webSQLEnabled { false };

#if ENABLE(ATTACHMENT_ELEMENT)
    bool m_isAttachmentElementEnabled { false };
#endif

    bool m_isWebRTCAudioLatencyAdaptationEnabled { true };

    bool m_isReadableByteStreamAPIEnabled { false };

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    bool m_accessibilityIsolatedTree { false };
#endif

#if ENABLE(AX_THREAD_TEXT_APIS)
    bool m_accessibilityThreadTextApis { false };
#endif

    bool m_arePDFImagesEnabled { true };

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
    bool m_builtInNotificationsEnabled { false };
#endif

#if ENABLE(MODEL_ELEMENT)
    bool m_modelDocumentEnabled { false };
#endif

    friend class NeverDestroyed<DeprecatedGlobalSettings>;
};

} // namespace WebCore
