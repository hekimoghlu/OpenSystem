/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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

#include <wtf/Compiler.h>

#import "AuxiliaryProcess.h"
#import "WebKit2Initialize.h"
#import <JavaScriptCore/ExecutableAllocator.h>
#import <wtf/OSObjectPtr.h>
#import <wtf/WTFProcess.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>

#if !USE(RUNNINGBOARD)
#import <wtf/spi/darwin/XPCSPI.h>
#endif

// FIXME: This should be moved to an SPI header.
#if USE(APPLE_INTERNAL_SDK)
#include <os/voucher_private.h>
#else
extern "C" OS_NOTHROW void voucher_replace_default_voucher(void);
#endif

#define WEBCONTENT_SERVICE_INITIALIZER WebContentServiceInitializer
#define NETWORK_SERVICE_INITIALIZER NetworkServiceInitializer
#define GPU_SERVICE_INITIALIZER GPUServiceInitializer
#define MODEL_SERVICE_INITIALIZER ModelServiceInitializer

namespace WebKit {

class XPCServiceInitializerDelegate {
public:
    XPCServiceInitializerDelegate(OSObjectPtr<xpc_connection_t>, xpc_object_t initializerMessage);

    virtual ~XPCServiceInitializerDelegate();

    virtual bool checkEntitlements();

    virtual bool getConnectionIdentifier(IPC::Connection::Identifier& identifier);
    virtual bool getProcessIdentifier(std::optional<WebCore::ProcessIdentifier>&);
    virtual bool getClientIdentifier(String& clientIdentifier);
    virtual bool getClientBundleIdentifier(String& clientBundleIdentifier);
    virtual bool getClientProcessName(String& clientProcessName);
    virtual bool getClientSDKAlignedBehaviors(SDKAlignedBehaviors&);
    virtual bool getExtraInitializationData(HashMap<String, String>& extraInitializationData);

protected:
    bool hasEntitlement(ASCIILiteral entitlement);
    bool isClientSandboxed();

    OSObjectPtr<xpc_connection_t> m_connection;
    xpc_object_t m_initializerMessage;
};

template<typename XPCServiceType>
void initializeAuxiliaryProcess(AuxiliaryProcessInitializationParameters&& parameters)
{
    XPCServiceType::singleton().initialize(WTFMove(parameters));
}

#if !USE(RUNNINGBOARD)
void setOSTransaction(OSObjectPtr<os_transaction_t>&&);
#endif

enum class EnableLockdownMode: bool { No, Yes };

void setJSCOptions(xpc_object_t initializerMessage, EnableLockdownMode, bool isWebContentProcess);

template<typename XPCServiceType, typename XPCServiceInitializerDelegateType, bool isWebContentProcess = false>
void XPCServiceInitializer(OSObjectPtr<xpc_connection_t> connection, xpc_object_t initializerMessage)
{
    XPCServiceInitializerDelegateType delegate(WTFMove(connection), initializerMessage);

    // We don't want XPC to be in charge of whether the process should be terminated or not,
    // so ensure that we have an outstanding transaction here. This is not needed when using
    // RunningBoard because the UIProcess takes process assertions on behalf of its child processes.
#if !USE(RUNNINGBOARD)
    setOSTransaction(adoptOSObject(os_transaction_create("WebKit XPC Service")));
#endif

    AuxiliaryProcessInitializationParameters parameters;

    if (!delegate.getExtraInitializationData(parameters.extraInitializationData))
        exitProcess(EXIT_FAILURE);

    if (isWebContentProcess)
        JSC::Options::machExceptionHandlerSandboxPolicy = JSC::Options::SandboxPolicy::Allow;
    if (initializerMessage) {
        bool enableLockdownMode = parameters.extraInitializationData.get<HashTranslatorASCIILiteral>("enable-lockdown-mode"_s) == "1"_s;
        setJSCOptions(initializerMessage, enableLockdownMode ? EnableLockdownMode::Yes : EnableLockdownMode::No, isWebContentProcess);
    }

    // InitializeWebKit2() calls linkedOnOrAfterSDKWithBehavior(), so SDK-aligned behaviors must be
    // configured beforehand.
    SDKAlignedBehaviors clientSDKAlignedBehaviors;
    delegate.getClientSDKAlignedBehaviors(clientSDKAlignedBehaviors);
    setSDKAlignedBehaviors(clientSDKAlignedBehaviors);

    // computeSDKAlignedBehaviors() asserts that it is not called in an auxiliary process, so
    // setAuxiliaryProcessType() should be called before the first call to
    // linkedOnOrAfterSDKWithBehavior() to ensure the assertion will catch bugs where
    // setSDKAlignedBehaviors() isn't called at the right time.
    parameters.processType = XPCServiceType::processType;
    setAuxiliaryProcessType(parameters.processType);

    InitializeWebKit2();

    if (!delegate.checkEntitlements())
        exitProcess(EXIT_FAILURE);

    if (!delegate.getConnectionIdentifier(parameters.connectionIdentifier))
        exitProcess(EXIT_FAILURE);

    if (!delegate.getClientIdentifier(parameters.clientIdentifier))
        exitProcess(EXIT_FAILURE);

    // The host process may not have a bundle identifier (e.g. a command line app), so don't require one.
    delegate.getClientBundleIdentifier(parameters.clientBundleIdentifier);

    std::optional<WebCore::ProcessIdentifier> processIdentifier;
    if (!delegate.getProcessIdentifier(processIdentifier))
        exitProcess(EXIT_FAILURE);
    parameters.processIdentifier = *processIdentifier;

    if (!delegate.getClientProcessName(parameters.uiProcessName))
        exitProcess(EXIT_FAILURE);

    // Set the task default voucher to the current value (as propagated by XPC).
    voucher_replace_default_voucher();

#if HAVE(QOS_CLASSES)
    if (parameters.extraInitializationData.contains("always-runs-at-background-priority"_s))
        Thread::setGlobalMaxQOSClass(QOS_CLASS_UTILITY);
#endif

    initializeAuxiliaryProcess<XPCServiceType>(WTFMove(parameters));
}

int XPCServiceMain(int, const char**);
void XPCServiceEventHandler(xpc_connection_t peer);
void XPCServiceExit();

} // namespace WebKit
