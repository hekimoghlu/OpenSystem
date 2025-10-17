/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#if (ENABLE(WEBKIT2) && (NESTED_MASTER_CONDITION || MASTER_OR && MASTER_AND))
#include "TestWithLegacyReceiver.h"

#include "ArgumentCoders.h" // NOLINT
#include "Connection.h" // NOLINT
#include "Decoder.h" // NOLINT
#if ENABLE(DEPRECATED_FEATURE) || ENABLE(FEATURE_FOR_TESTING)
#include "DummyType.h" // NOLINT
#endif
#if PLATFORM(MAC)
#include "GestureTypes.h" // NOLINT
#endif
#include "HandleMessage.h" // NOLINT
#include "Plugin.h" // NOLINT
#include "TestWithLegacyReceiverMessages.h" // NOLINT
#include "WebPreferencesStore.h" // NOLINT
#if (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION && SOME_OTHER_MESSAGE_CONDITION)) || (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION || SOME_OTHER_MESSAGE_CONDITION))
#include "WebTouchEvent.h" // NOLINT
#endif
#if PLATFORM(MAC)
#include <WebCore/KeyboardEvent.h> // NOLINT
#endif
#include <WebCore/PlatformLayerIdentifier.h> // NOLINT
#include <WebCore/PluginData.h> // NOLINT
#include <utility> // NOLINT
#include <wtf/HashMap.h> // NOLINT
#if PLATFORM(MAC)
#include <wtf/MachSendRight.h> // NOLINT
#endif
#if PLATFORM(MAC)
#include <wtf/OptionSet.h> // NOLINT
#endif
#include <wtf/Vector.h> // NOLINT
#include <wtf/text/WTFString.h> // NOLINT

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithLegacyReceiver::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::LoadURL::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::LoadURL>(connection, decoder, this, &TestWithLegacyReceiver::loadURL);
#if ENABLE(TOUCH_EVENTS)
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::LoadSomething::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::LoadSomething>(connection, decoder, this, &TestWithLegacyReceiver::loadSomething);
#endif
#if (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION || SOME_OTHER_MESSAGE_CONDITION))
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::TouchEvent::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::TouchEvent>(connection, decoder, this, &TestWithLegacyReceiver::touchEvent);
#endif
#if (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION && SOME_OTHER_MESSAGE_CONDITION))
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::AddEvent::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::AddEvent>(connection, decoder, this, &TestWithLegacyReceiver::addEvent);
#endif
#if ENABLE(TOUCH_EVENTS)
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::LoadSomethingElse::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::LoadSomethingElse>(connection, decoder, this, &TestWithLegacyReceiver::loadSomethingElse);
#endif
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::DidReceivePolicyDecision::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::DidReceivePolicyDecision>(connection, decoder, this, &TestWithLegacyReceiver::didReceivePolicyDecision);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::Close::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::Close>(connection, decoder, this, &TestWithLegacyReceiver::close);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::PreferencesDidChange::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::PreferencesDidChange>(connection, decoder, this, &TestWithLegacyReceiver::preferencesDidChange);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::SendDoubleAndFloat::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::SendDoubleAndFloat>(connection, decoder, this, &TestWithLegacyReceiver::sendDoubleAndFloat);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::SendInts::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::SendInts>(connection, decoder, this, &TestWithLegacyReceiver::sendInts);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::CreatePlugin::name())
        return IPC::handleMessageAsync<Messages::TestWithLegacyReceiver::CreatePlugin>(connection, decoder, this, &TestWithLegacyReceiver::createPlugin);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::RunJavaScriptAlert::name())
        return IPC::handleMessageAsync<Messages::TestWithLegacyReceiver::RunJavaScriptAlert>(connection, decoder, this, &TestWithLegacyReceiver::runJavaScriptAlert);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::GetPlugins::name())
        return IPC::handleMessageAsync<Messages::TestWithLegacyReceiver::GetPlugins>(connection, decoder, this, &TestWithLegacyReceiver::getPlugins);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::TestParameterAttributes::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::TestParameterAttributes>(connection, decoder, this, &TestWithLegacyReceiver::testParameterAttributes);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::TemplateTest::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::TemplateTest>(connection, decoder, this, &TestWithLegacyReceiver::templateTest);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::SetVideoLayerID::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::SetVideoLayerID>(connection, decoder, this, &TestWithLegacyReceiver::setVideoLayerID);
#if PLATFORM(MAC)
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::DidCreateWebProcessConnection::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::DidCreateWebProcessConnection>(connection, decoder, this, &TestWithLegacyReceiver::didCreateWebProcessConnection);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::InterpretKeyEvent::name())
        return IPC::handleMessageAsync<Messages::TestWithLegacyReceiver::InterpretKeyEvent>(connection, decoder, this, &TestWithLegacyReceiver::interpretKeyEvent);
#endif
#if ENABLE(DEPRECATED_FEATURE)
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::DeprecatedOperation::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::DeprecatedOperation>(connection, decoder, this, &TestWithLegacyReceiver::deprecatedOperation);
#endif
#if ENABLE(FEATURE_FOR_TESTING)
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::ExperimentalOperation::name())
        return IPC::handleMessage<Messages::TestWithLegacyReceiver::ExperimentalOperation>(connection, decoder, this, &TestWithLegacyReceiver::experimentalOperation);
#endif
    UNUSED_PARAM(connection);
    RELEASE_LOG_ERROR(IPC, "Unhandled message %s to %" PRIu64, IPC::description(decoder.messageName()).characters(), decoder.destinationID());
    decoder.markInvalid();
}

bool TestWithLegacyReceiver::didReceiveSyncMessage(IPC::Connection& connection, IPC::Decoder& decoder, UniqueRef<IPC::Encoder>& replyEncoder)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::GetPluginProcessConnection::name())
        return IPC::handleMessageSynchronous<Messages::TestWithLegacyReceiver::GetPluginProcessConnection>(connection, decoder, replyEncoder, this, &TestWithLegacyReceiver::getPluginProcessConnection);
    if (decoder.messageName() == Messages::TestWithLegacyReceiver::TestMultipleAttributes::name())
        return IPC::handleMessageSynchronous<Messages::TestWithLegacyReceiver::TestMultipleAttributes>(connection, decoder, replyEncoder, this, &TestWithLegacyReceiver::testMultipleAttributes);
    UNUSED_PARAM(connection);
    UNUSED_PARAM(replyEncoder);
    RELEASE_LOG_ERROR(IPC, "Unhandled synchronous message %s to %" PRIu64, description(decoder.messageName()).characters(), decoder.destinationID());
    decoder.markInvalid();
    return false;
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_LoadURL>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::LoadURL::Arguments>(globalObject, decoder);
}
#if ENABLE(TOUCH_EVENTS)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_LoadSomething>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::LoadSomething::Arguments>(globalObject, decoder);
}
#endif
#if (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION || SOME_OTHER_MESSAGE_CONDITION))
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_TouchEvent>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::TouchEvent::Arguments>(globalObject, decoder);
}
#endif
#if (ENABLE(TOUCH_EVENTS) && (NESTED_MESSAGE_CONDITION && SOME_OTHER_MESSAGE_CONDITION))
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_AddEvent>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::AddEvent::Arguments>(globalObject, decoder);
}
#endif
#if ENABLE(TOUCH_EVENTS)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_LoadSomethingElse>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::LoadSomethingElse::Arguments>(globalObject, decoder);
}
#endif
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_DidReceivePolicyDecision>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::DidReceivePolicyDecision::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_Close>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::Close::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_PreferencesDidChange>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::PreferencesDidChange::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_SendDoubleAndFloat>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::SendDoubleAndFloat::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_SendInts>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::SendInts::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_CreatePlugin>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::CreatePlugin::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithLegacyReceiver_CreatePlugin>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::CreatePlugin::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_RunJavaScriptAlert>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::RunJavaScriptAlert::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithLegacyReceiver_RunJavaScriptAlert>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::RunJavaScriptAlert::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_GetPlugins>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::GetPlugins::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithLegacyReceiver_GetPlugins>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::GetPlugins::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_GetPluginProcessConnection>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::GetPluginProcessConnection::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithLegacyReceiver_GetPluginProcessConnection>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::GetPluginProcessConnection::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_TestMultipleAttributes>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::TestMultipleAttributes::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithLegacyReceiver_TestMultipleAttributes>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::TestMultipleAttributes::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_TestParameterAttributes>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::TestParameterAttributes::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_TemplateTest>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::TemplateTest::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_SetVideoLayerID>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::SetVideoLayerID::Arguments>(globalObject, decoder);
}
#if PLATFORM(MAC)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_DidCreateWebProcessConnection>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::DidCreateWebProcessConnection::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_InterpretKeyEvent>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::InterpretKeyEvent::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithLegacyReceiver_InterpretKeyEvent>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::InterpretKeyEvent::ReplyArguments>(globalObject, decoder);
}
#endif
#if ENABLE(DEPRECATED_FEATURE)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_DeprecatedOperation>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::DeprecatedOperation::Arguments>(globalObject, decoder);
}
#endif
#if ENABLE(FEATURE_FOR_TESTING)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithLegacyReceiver_ExperimentalOperation>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithLegacyReceiver::ExperimentalOperation::Arguments>(globalObject, decoder);
}
#endif

}

#endif


#endif // (ENABLE(WEBKIT2) && (NESTED_MASTER_CONDITION || MASTER_OR && MASTER_AND))
