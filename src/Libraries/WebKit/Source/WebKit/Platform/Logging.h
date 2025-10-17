/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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

#include <wtf/Assertions.h>
#include <wtf/text/WTFString.h>

#if __has_include("WebKitLogDefinitions.h")
#include "WebKitLogDefinitions.h"
#endif

#define COMMA() ,
#define OPTIONAL_ARGS(...) __VA_OPT__(COMMA()) __VA_ARGS__

#if ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)
#include "LogClient.h"

#define RELEASE_LOG_FORWARDABLE(category, logMessage, ...) do { \
    if (auto* client = downcast<LogClient>(WebCore::logClient().get())) \
        client->logMessage(__VA_ARGS__); \
    else \
        RELEASE_LOG(category, MESSAGE_##logMessage OPTIONAL_ARGS(__VA_ARGS__)); \
} while (0)

#define RELEASE_LOG_INFO_FORWARDABLE(category, logMessage, ...) do { \
    if (auto* client = downcast<LogClient>(WebCore::logClient().get())) \
        client->logMessage(__VA_ARGS__); \
    else \
        RELEASE_LOG_INFO(category, MESSAGE_##logMessage OPTIONAL_ARGS(__VA_ARGS__)); \
} while (0)

#define RELEASE_LOG_ERROR_FORWARDABLE(category, logMessage, ...) do { \
    if (auto* client = downcast<LogClient>(WebCore::logClient().get())) \
        client->logMessage(__VA_ARGS__); \
    else \
        RELEASE_LOG_ERROR(category, MESSAGE_##logMessage OPTIONAL_ARGS(__VA_ARGS__)); \
} while (0)

#define RELEASE_LOG_FAULT_FORWARDABLE(category, logMessage, ...) do { \
    if (auto* client = downcast<LogClient>(WebCore::logClient().get())) \
        client->logMessage(__VA_ARGS__); \
    else \
        RELEASE_LOG_FAULT(category, MESSAGE_##logMessage OPTIONAL_ARGS(__VA_ARGS__)); \
} while (0)
#else
#define RELEASE_LOG_FORWARDABLE(category, logMessage, ...) RELEASE_LOG(category, MESSAGE_##logMessage OPTIONAL_ARGS(__VA_ARGS__))
#define RELEASE_LOG_INFO_FORWARDABLE(category, logMessage, ...) RELEASE_LOG_INFO(category, MESSAGE_##logMessage OPTIONAL_ARGS(__VA_ARGS__))
#define RELEASE_LOG_ERROR_FORWARDABLE(category, logMessage, ...) RELEASE_LOG_ERROR(category, MESSAGE_##logMessage OPTIONAL_ARGS(__VA_ARGS__))
#define RELEASE_LOG_FAULT_FORWARDABLE(category, logMessage, ...) RELEASE_LOG_FAULT(category, MESSAGE_##logMessage OPTIONAL_ARGS(__VA_ARGS__))
#endif // ENABLE(LOGD_BLOCKING_IN_WEBCONTENT)

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

#ifndef LOG_CHANNEL_PREFIX
#define LOG_CHANNEL_PREFIX WebKit2Log
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define WEBKIT2_LOG_CHANNELS(M) \
    M(API) \
    M(ActivityState) \
    M(AdvancedPrivacyProtections) \
    M(AppSSO) \
    M(Animations) \
    M(Automation) \
    M(AutomationInteractions) \
    M(BackForward) \
    M(BackForwardCache) \
    M(CacheStorage) \
    M(ContentObservation) \
    M(ContentRuleLists) \
    M(ContextMenu) \
    M(DisplayLink) \
    M(DisplayLists) \
    M(DiskPersistency) \
    M(DragAndDrop) \
    M(EME) \
    M(Extensions) \
    M(Fullscreen) \
    M(Gamepad) \
    M(IPC) \
    M(IPCMessages) \
    M(ITPDebug) \
    M(IconDatabase) \
    M(Images) \
    M(ImageAnalysis) \
    M(IncrementalPDF) \
    M(IncrementalPDFVerbose) \
    M(IndexedDB) \
    M(Inspector) \
    M(KeyHandling) \
    M(Keychain) \
    M(Language) \
    M(Layers) \
    M(Layout) \
    M(Loading) \
    M(Media) \
    M(MemoryPressure) \
    M(ModelElement) \
    M(MouseHandling) \
    M(Network) \
    M(NetworkCache) \
    M(NetworkCacheSpeculativePreloading) \
    M(NetworkCacheStorage) \
    M(NetworkScheduling) \
    M(NetworkSession) \
    M(Notifications) \
    M(PDF) \
    M(PDFAsyncRendering) \
    M(PageLoadObserver) \
    M(PerformanceLogging) \
    M(Plugins) \
    M(Printing) \
    M(PrivateClickMeasurement) \
    M(Process) \
    M(ProcessCapabilities) \
    M(ProcessSuspension) \
    M(ProcessSwapping) \
    M(ProximityNetworking) \
    M(Push) \
    M(RemoteLayerBuffers) \
    M(RemoteLayerTree) \
    M(Resize) \
    M(ResourceLoadStatistics) \
    M(Sandbox) \
    M(ScrollAnimations) \
    M(Scrolling) \
    M(SecureCoding) \
    M(Selection) \
    M(ServiceWorker) \
    M(SessionState) \
    M(SharedDisplayLists) \
    M(SharedWorker) \
    M(SiteIsolation) \
    M(Storage) \
    M(StorageAPI) \
    M(SystemPreview) \
    M(Telephony) \
    M(TextInput) \
    M(TextInteraction) \
    M(Translation) \
    M(UIHitTesting) \
    M(ViewGestures) \
    M(ViewState) \
    M(ViewportSizing) \
    M(VirtualMemory) \
    M(VisibleRects) \
    M(WebAuthn) \
    M(WebGL) \
    M(WebRTC) \
    M(WheelEvents) \
    M(Worker) \
    M(XR) \

WEBKIT2_LOG_CHANNELS(DECLARE_LOG_CHANNEL)

#undef DECLARE_LOG_CHANNEL

#ifdef __cplusplus
}
#endif

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED
