/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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

#include <span>
#include <wtf/Forward.h>
#include <wtf/OptionSet.h>

typedef const struct OpaqueJSContext* JSContextRef;
typedef struct OpaqueJSString* JSStringRef;
typedef struct OpaqueJSValue* JSObjectRef;

#if PLATFORM(COCOA)
#define TEST_SUPPORT_EXPORT WTF_EXPORT_PRIVATE
#else
#define TEST_SUPPORT_EXPORT
#endif

namespace WebCore {
class LocalFrame;
enum class ParserContentPolicy : uint8_t;
}

namespace WebCoreTestSupport {

TEST_SUPPORT_EXPORT void initializeNames();

TEST_SUPPORT_EXPORT void injectInternalsObject(JSContextRef);
TEST_SUPPORT_EXPORT void resetInternalsObject(JSContextRef);
TEST_SUPPORT_EXPORT void monitorWheelEvents(WebCore::LocalFrame&, bool clearLatchingState);
TEST_SUPPORT_EXPORT void setWheelEventMonitorTestCallbackAndStartMonitoring(bool expectWheelEndOrCancel, bool expectMomentumEnd, WebCore::LocalFrame&, JSContextRef, JSObjectRef);
TEST_SUPPORT_EXPORT void clearWheelEventTestMonitor(WebCore::LocalFrame&);

TEST_SUPPORT_EXPORT void setLogChannelToAccumulate(const String& name);
TEST_SUPPORT_EXPORT void clearAllLogChannelsToAccumulate();
TEST_SUPPORT_EXPORT void initializeLogChannelsIfNecessary();
TEST_SUPPORT_EXPORT void setAllowsAnySSLCertificate(bool);
TEST_SUPPORT_EXPORT bool allowsAnySSLCertificate();
TEST_SUPPORT_EXPORT void setLinkedOnOrAfterEverythingForTesting();

TEST_SUPPORT_EXPORT void installMockGamepadProvider();
TEST_SUPPORT_EXPORT void connectMockGamepad(unsigned index);
TEST_SUPPORT_EXPORT void disconnectMockGamepad(unsigned index);
TEST_SUPPORT_EXPORT void setMockGamepadDetails(unsigned index, const String& gamepadID, const String& mapping, unsigned axisCount, unsigned buttonCount, bool supportsDualRumble);
TEST_SUPPORT_EXPORT void setMockGamepadAxisValue(unsigned index, unsigned axisIndex, double value);
TEST_SUPPORT_EXPORT void setMockGamepadButtonValue(unsigned index, unsigned buttonIndex, double value);

TEST_SUPPORT_EXPORT void setupNewlyCreatedServiceWorker(uint64_t serviceWorkerIdentifier);
    
TEST_SUPPORT_EXPORT void setAdditionalSupportedImageTypesForTesting(const String&);

#if ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY)
#if ENABLE(JIT_OPERATION_DISASSEMBLY)
TEST_SUPPORT_EXPORT void populateDisassemblyLabels();
#else
inline void populateDisassemblyLabels() { }
#endif

#if ENABLE(JIT_OPERATION_VALIDATION)
TEST_SUPPORT_EXPORT void populateJITOperations();
#else
inline void populateJITOperations() { populateDisassemblyLabels(); }
#endif

#else
inline void populateJITOperations() { }
#endif // ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY)

TEST_SUPPORT_EXPORT bool testDocumentFragmentParseXML(const String&, OptionSet<WebCore::ParserContentPolicy>);

#if ENABLE(WEB_AUDIO)
TEST_SUPPORT_EXPORT void testSincResamplerProcessBuffer(std::span<const float> source, std::span<float> destination, double scaleFactor);
#endif

} // namespace WebCoreTestSupport
