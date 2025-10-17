/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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

#include <algorithm>
#include <wtf/EnumTraits.h>
#include <wtf/text/ASCIILiteral.h>

namespace IPC {

enum class ReceiverName : uint8_t {
    TestWithCVPixelBuffer = 1
    , TestWithDeferSendingOption = 2
    , TestWithDispatchedFromAndTo = 3
    , TestWithEnabledBy = 4
    , TestWithEnabledByAndConjunction = 5
    , TestWithEnabledByOrConjunction = 6
    , TestWithIfMessage = 7
    , TestWithImageData = 8
    , TestWithLegacyReceiver = 9
    , TestWithMultiLineExtendedAttributes = 10
    , TestWithSemaphore = 11
    , TestWithStream = 12
    , TestWithStreamBatched = 13
    , TestWithStreamBuffer = 14
    , TestWithStreamServerConnectionHandle = 15
    , TestWithSuperclass = 16
    , TestWithSuperclassAndWantsAsyncDispatch = 17
    , TestWithSuperclassAndWantsDispatch = 18
    , TestWithValidator = 19
    , TestWithWantsAsyncDispatch = 20
    , TestWithWantsDispatch = 21
    , TestWithWantsDispatchNoSyncMessages = 22
    , TestWithoutAttributes = 23
    , TestWithoutUsingIPCConnection = 24
    , IPC = 25
    , AsyncReply = 26
    , Invalid = 27
};

enum class MessageName : uint16_t {
#if USE(AVFOUNDATION)
    TestWithCVPixelBuffer_ReceiveCVPixelBuffer,
    TestWithCVPixelBuffer_SendCVPixelBuffer,
#endif
    TestWithDeferSendingOption_MultipleIndices,
    TestWithDeferSendingOption_NoIndices,
    TestWithDeferSendingOption_NoOptions,
    TestWithDeferSendingOption_OneIndex,
    TestWithDispatchedFromAndTo_AlwaysEnabled,
    TestWithEnabledByAndConjunction_AlwaysEnabled,
    TestWithEnabledByOrConjunction_AlwaysEnabled,
    TestWithEnabledBy_AlwaysEnabled,
    TestWithEnabledBy_ConditionallyEnabled,
    TestWithEnabledBy_ConditionallyEnabledAnd,
    TestWithEnabledBy_ConditionallyEnabledOr,
#if PLATFORM(COCOA) || PLATFORM(GTK)
    TestWithIfMessage_LoadURL,
#endif
    TestWithImageData_ReceiveImageData,
    TestWithImageData_SendImageData,
#if (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION && SOME_OTHER_MESSAGE_CONDITION))
    TestWithLegacyReceiver_AddEvent,
#endif
    TestWithLegacyReceiver_Close,
    TestWithLegacyReceiver_CreatePlugin,
#if ENABLE(DEPRECATED_FEATURE)
    TestWithLegacyReceiver_DeprecatedOperation,
#endif
#if PLATFORM(MAC)
    TestWithLegacyReceiver_DidCreateWebProcessConnection,
#endif
    TestWithLegacyReceiver_DidReceivePolicyDecision,
#if ENABLE(FEATURE_FOR_TESTING)
    TestWithLegacyReceiver_ExperimentalOperation,
#endif
    TestWithLegacyReceiver_GetPlugins,
#if PLATFORM(MAC)
    TestWithLegacyReceiver_InterpretKeyEvent,
#endif
#if ENABLE(TOUCH_EVENTS)
    TestWithLegacyReceiver_LoadSomething,
    TestWithLegacyReceiver_LoadSomethingElse,
#endif
    TestWithLegacyReceiver_LoadURL,
    TestWithLegacyReceiver_PreferencesDidChange,
    TestWithLegacyReceiver_RunJavaScriptAlert,
    TestWithLegacyReceiver_SendDoubleAndFloat,
    TestWithLegacyReceiver_SendInts,
    TestWithLegacyReceiver_SetVideoLayerID,
    TestWithLegacyReceiver_TemplateTest,
    TestWithLegacyReceiver_TestParameterAttributes,
#if (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION || SOME_OTHER_MESSAGE_CONDITION))
    TestWithLegacyReceiver_TouchEvent,
#endif
    TestWithMultiLineExtendedAttributes_AlwaysEnabled,
    TestWithSemaphore_ReceiveSemaphore,
    TestWithSemaphore_SendSemaphore,
    TestWithStreamBatched_SendString,
    TestWithStreamBuffer_SendStreamBuffer,
    TestWithStreamServerConnectionHandle_SendStreamServerConnection,
    TestWithStream_CallWithIdentifier,
#if PLATFORM(COCOA)
    TestWithStream_SendMachSendRight,
#endif
    TestWithStream_SendString,
    TestWithStream_SendStringAsync,
    TestWithSuperclassAndWantsAsyncDispatch_LoadURL,
    TestWithSuperclassAndWantsDispatch_LoadURL,
    TestWithSuperclass_LoadURL,
#if ENABLE(TEST_FEATURE)
    TestWithSuperclass_TestAsyncMessage,
    TestWithSuperclass_TestAsyncMessageWithConnection,
    TestWithSuperclass_TestAsyncMessageWithMultipleArguments,
    TestWithSuperclass_TestAsyncMessageWithNoArguments,
#endif
    TestWithValidator_AlwaysEnabled,
    TestWithValidator_EnabledIfPassValidation,
    TestWithValidator_EnabledIfSomeFeatureEnabledAndPassValidation,
    TestWithWantsAsyncDispatch_TestMessage,
    TestWithWantsDispatchNoSyncMessages_TestMessage,
    TestWithWantsDispatch_TestMessage,
#if (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION && SOME_OTHER_MESSAGE_CONDITION))
    TestWithoutAttributes_AddEvent,
#endif
    TestWithoutAttributes_Close,
    TestWithoutAttributes_CreatePlugin,
#if ENABLE(DEPRECATED_FEATURE)
    TestWithoutAttributes_DeprecatedOperation,
#endif
#if PLATFORM(MAC)
    TestWithoutAttributes_DidCreateWebProcessConnection,
#endif
    TestWithoutAttributes_DidReceivePolicyDecision,
#if ENABLE(FEATURE_FOR_TESTING)
    TestWithoutAttributes_ExperimentalOperation,
#endif
    TestWithoutAttributes_GetPlugins,
#if PLATFORM(MAC)
    TestWithoutAttributes_InterpretKeyEvent,
#endif
#if ENABLE(TOUCH_EVENTS)
    TestWithoutAttributes_LoadSomething,
    TestWithoutAttributes_LoadSomethingElse,
#endif
    TestWithoutAttributes_LoadURL,
    TestWithoutAttributes_PreferencesDidChange,
    TestWithoutAttributes_RunJavaScriptAlert,
    TestWithoutAttributes_SendDoubleAndFloat,
    TestWithoutAttributes_SendInts,
    TestWithoutAttributes_SetVideoLayerID,
    TestWithoutAttributes_TemplateTest,
    TestWithoutAttributes_TestParameterAttributes,
#if (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION || SOME_OTHER_MESSAGE_CONDITION))
    TestWithoutAttributes_TouchEvent,
#endif
    TestWithoutUsingIPCConnection_MessageWithArgument,
    TestWithoutUsingIPCConnection_MessageWithArgumentAndEmptyReply,
    TestWithoutUsingIPCConnection_MessageWithArgumentAndReplyWithArgument,
    TestWithoutUsingIPCConnection_MessageWithoutArgument,
    TestWithoutUsingIPCConnection_MessageWithoutArgumentAndEmptyReply,
    TestWithoutUsingIPCConnection_MessageWithoutArgumentAndReplyWithArgument,
    CancelSyncMessageReply,
#if PLATFORM(COCOA)
    InitializeConnection,
#endif
    LegacySessionState,
    ProcessOutOfStreamMessage,
    SetStreamDestinationID,
    SyncMessageReply,
#if USE(AVFOUNDATION)
    TestWithCVPixelBuffer_ReceiveCVPixelBufferReply,
#endif
    TestWithImageData_ReceiveImageDataReply,
    TestWithLegacyReceiver_CreatePluginReply,
    TestWithLegacyReceiver_GetPluginsReply,
#if PLATFORM(MAC)
    TestWithLegacyReceiver_InterpretKeyEventReply,
#endif
    TestWithLegacyReceiver_RunJavaScriptAlertReply,
    TestWithSemaphore_ReceiveSemaphoreReply,
    TestWithStream_CallWithIdentifierReply,
    TestWithStream_SendStringAsyncReply,
#if ENABLE(TEST_FEATURE)
    TestWithSuperclass_TestAsyncMessageReply,
    TestWithSuperclass_TestAsyncMessageWithConnectionReply,
    TestWithSuperclass_TestAsyncMessageWithMultipleArgumentsReply,
    TestWithSuperclass_TestAsyncMessageWithNoArgumentsReply,
#endif
    TestWithoutAttributes_CreatePluginReply,
    TestWithoutAttributes_GetPluginsReply,
#if PLATFORM(MAC)
    TestWithoutAttributes_InterpretKeyEventReply,
#endif
    TestWithoutAttributes_RunJavaScriptAlertReply,
    TestWithoutUsingIPCConnection_MessageWithArgumentAndEmptyReplyReply,
    TestWithoutUsingIPCConnection_MessageWithArgumentAndReplyWithArgumentReply,
    TestWithoutUsingIPCConnection_MessageWithoutArgumentAndEmptyReplyReply,
    TestWithoutUsingIPCConnection_MessageWithoutArgumentAndReplyWithArgumentReply,
    FirstSynchronous,
    LastAsynchronous = FirstSynchronous - 1,
    TestWithLegacyReceiver_GetPluginProcessConnection,
    TestWithLegacyReceiver_TestMultipleAttributes,
#if PLATFORM(COCOA)
    TestWithStream_ReceiveMachSendRight,
    TestWithStream_SendAndReceiveMachSendRight,
#endif
    TestWithStream_SendStringSync,
    TestWithSuperclassAndWantsAsyncDispatch_TestSyncMessage,
    TestWithSuperclassAndWantsDispatch_TestSyncMessage,
    TestWithSuperclass_TestSyncMessage,
    TestWithSuperclass_TestSynchronousMessage,
    TestWithWantsAsyncDispatch_TestSyncMessage,
    TestWithWantsDispatch_TestSyncMessage,
    TestWithoutAttributes_GetPluginProcessConnection,
    TestWithoutAttributes_TestMultipleAttributes,
    WrappedAsyncMessageForTesting,
    Count,
    Invalid = Count,
    Last = Count - 1
};

namespace Detail {
struct MessageDescription {
    ASCIILiteral description;
    ReceiverName receiverName;
    bool messageAllowedWhenWaitingForSyncReply : 1;
    bool messageAllowedWhenWaitingForUnboundedSyncReply : 1;
};

using MessageDescriptionsArray = std::array<MessageDescription, static_cast<size_t>(MessageName::Count) + 1>;
extern const MessageDescriptionsArray messageDescriptions;

}

inline ReceiverName receiverName(MessageName messageName)
{
    messageName = std::min(messageName, MessageName::Last);
    return Detail::messageDescriptions[static_cast<size_t>(messageName)].receiverName;
}

inline ASCIILiteral description(MessageName messageName)
{
    messageName = std::min(messageName, MessageName::Last);
    return Detail::messageDescriptions[static_cast<size_t>(messageName)].description;
}

inline bool messageAllowedWhenWaitingForSyncReply(MessageName messageName)
{
    messageName = std::min(messageName, MessageName::Last);
    return Detail::messageDescriptions[static_cast<size_t>(messageName)].messageAllowedWhenWaitingForSyncReply;
}

inline bool messageAllowedWhenWaitingForUnboundedSyncReply(MessageName messageName)
{
    messageName = std::min(messageName, MessageName::Last);
    return Detail::messageDescriptions[static_cast<size_t>(messageName)].messageAllowedWhenWaitingForUnboundedSyncReply;
}

constexpr bool messageIsSync(MessageName name)
{
    return name >= MessageName::FirstSynchronous;
}

} // namespace IPC

namespace WTF {

template<> constexpr bool isValidEnum<IPC::MessageName>(std::underlying_type_t<IPC::MessageName> messageName)
{
    return messageName <= WTF::enumToUnderlyingType(IPC::MessageName::Last);
}

} // namespace WTF
