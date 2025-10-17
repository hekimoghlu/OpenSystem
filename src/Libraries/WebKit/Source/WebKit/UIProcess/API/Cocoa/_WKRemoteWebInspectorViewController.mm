/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#import "config.h"
#import "_WKRemoteWebInspectorViewControllerInternal.h"

#if PLATFORM(MAC)

#import "APIDebuggableInfo.h"
#import "APIInspectorConfiguration.h"
#import "DebuggableInfoData.h"
#import "RemoteWebInspectorUIProxy.h"
#import "WKWebViewInternal.h"
#import "_WKInspectorConfigurationInternal.h"
#import "_WKInspectorDebuggableInfoInternal.h"
#import <wtf/TZoneMallocInlines.h>

#if ENABLE(INSPECTOR_EXTENSIONS)
#import "APIInspectorExtension.h"
#import "WKError.h"
#import "WebInspectorUIExtensionControllerProxy.h"
#import "_WKInspectorExtensionInternal.h"
#import <wtf/BlockPtr.h>
#endif


@interface _WKRemoteWebInspectorViewController ()
- (void)sendMessageToBackend:(NSString *)message;
- (void)closeFromFrontend;
@end

namespace WebKit {

class _WKRemoteWebInspectorUIProxyClient final : public RemoteWebInspectorUIProxyClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(_WKRemoteWebInspectorUIProxyClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(_WKRemoteWebInspectorUIProxyClient);
public:
    _WKRemoteWebInspectorUIProxyClient(_WKRemoteWebInspectorViewController *controller)
        : m_controller(controller)
    {
    }

    virtual ~_WKRemoteWebInspectorUIProxyClient()
    {
    }

    void sendMessageToBackend(const String& message) override
    {
        [m_controller sendMessageToBackend:message];
    }

    void closeFromFrontend() override
    {
        [m_controller closeFromFrontend];
    }
    
    Ref<API::InspectorConfiguration> configurationForRemoteInspector(RemoteWebInspectorUIProxy& inspector) override
    {
        return downcast<API::InspectorConfiguration>([m_controller.configuration _apiObject]);
    }

private:
    _WKRemoteWebInspectorViewController *m_controller;
};

} // namespace WebKit

@implementation _WKRemoteWebInspectorViewController {
    std::unique_ptr<WebKit::_WKRemoteWebInspectorUIProxyClient> m_remoteInspectorClient;
    _WKInspectorConfiguration *_configuration;
}

- (instancetype)initWithConfiguration:(_WKInspectorConfiguration *)configuration
{
    if (!(self = [super init]))
        return nil;

    _configuration = [configuration copy];

    m_remoteInspectorProxy = WebKit::RemoteWebInspectorUIProxy::create();
    m_remoteInspectorClient = makeUnique<WebKit::_WKRemoteWebInspectorUIProxyClient>(self);
    m_remoteInspectorProxy->setClient(m_remoteInspectorClient.get());

    return self;
}

- (void)dealloc
{
    m_remoteInspectorProxy->setClient(nullptr);
    [_configuration release];
    [super dealloc];
}

- (NSWindow *)window
{
    return m_remoteInspectorProxy->window();
}

- (WKWebView *)webView
{
    return m_remoteInspectorProxy->webView();
}

- (void)loadForDebuggable:(_WKInspectorDebuggableInfo *)debuggableInfo backendCommandsURL:(NSURL *)backendCommandsURL
{
    m_remoteInspectorProxy->initialize(downcast<API::DebuggableInfo>([debuggableInfo _apiObject]), backendCommandsURL.absoluteString);
}

- (void)close
{
    m_remoteInspectorProxy->closeFromBackend();
}

- (void)show
{
    m_remoteInspectorProxy->show();
}

- (void)sendMessageToFrontend:(NSString *)message
{
    m_remoteInspectorProxy->sendMessageToFrontend(message);
}

// MARK: RemoteWebInspectorUIProxyClient methods

- (void)sendMessageToBackend:(NSString *)message
{
    if (_delegate && [_delegate respondsToSelector:@selector(inspectorViewController:sendMessageToBackend:)])
        [_delegate inspectorViewController:self sendMessageToBackend:message];
}

- (void)closeFromFrontend
{
    if (_delegate && [_delegate respondsToSelector:@selector(inspectorViewControllerInspectorDidClose:)])
        [_delegate inspectorViewControllerInspectorDidClose:self];
}

// MARK: _WKRemoteWebInspectorViewControllerPrivate methods

- (void)_setDiagnosticLoggingDelegate:(id<_WKDiagnosticLoggingDelegate>)delegate
{
    self.webView._diagnosticLoggingDelegate = delegate;
    m_remoteInspectorProxy->setDiagnosticLoggingAvailable(!!delegate);
}

// MARK: _WKInspectorExtensionHost methods

- (WKWebView *)extensionHostWebView
{
    return self.webView;
}

- (void)registerExtensionWithID:(NSString *)extensionID extensionBundleIdentifier:(NSString *)extensionBundleIdentifier displayName:(NSString *)displayName completionHandler:(void(^)(NSError *, _WKInspectorExtension *))completionHandler
{
#if ENABLE(INSPECTOR_EXTENSIONS)
    // If this method is called prior to creating a frontend with -loadForDebuggable:backendCommandsURL:, it will not succeed.
    if (!m_remoteInspectorProxy->extensionController()) {
        completionHandler([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(Inspector::ExtensionError::InvalidRequest) }], nil);
        return;
    }

    m_remoteInspectorProxy->extensionController()->registerExtension(extensionID, extensionBundleIdentifier, displayName, [protectedExtensionID = retainPtr(extensionID), protectedSelf = retainPtr(self), capturedBlock = makeBlockPtr(completionHandler)] (Expected<RefPtr<API::InspectorExtension>, Inspector::ExtensionError> result) mutable {
        if (!result) {
            capturedBlock([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(result.error()) }], nil);
            return;
        }

        capturedBlock(nil, wrapper(result.value()));
    });
#else
    completionHandler([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:nil], nil);
#endif
}

- (void)unregisterExtension:(_WKInspectorExtension *)extension completionHandler:(void(^)(NSError *))completionHandler
{
#if ENABLE(INSPECTOR_EXTENSIONS)
    // If this method is called prior to creating a frontend with -loadForDebuggable:backendCommandsURL:, it will not succeed.
    if (!m_remoteInspectorProxy->extensionController()) {
        completionHandler([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(Inspector::ExtensionError::InvalidRequest) }]);
        return;
    }
    m_remoteInspectorProxy->extensionController()->unregisterExtension(extension.extensionID, [protectedSelf = retainPtr(self), capturedBlock = makeBlockPtr(completionHandler)] (Expected<void, Inspector::ExtensionError> result) mutable {
        if (!result) {
            capturedBlock([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(result.error()) }]);
            return;
        }

        capturedBlock(nil);
    });
#else
    completionHandler([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:nil]);
#endif
}

- (void)showConsole
{
    m_remoteInspectorProxy->showConsole();
}

- (void)showResources
{
    m_remoteInspectorProxy->showResources();
}

- (void)showExtensionTabWithIdentifier:(NSString *)extensionTabIdentifier completionHandler:(void(^)(NSError *))completionHandler
{
#if ENABLE(INSPECTOR_EXTENSIONS)
    // It is an error to call this method prior to creating a frontend (i.e., with -connect or -show).
    if (!m_remoteInspectorProxy->extensionController()) {
        completionHandler([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(Inspector::ExtensionError::InvalidRequest)}]);
        return;
    }

    m_remoteInspectorProxy->extensionController()->showExtensionTab(extensionTabIdentifier, [protectedSelf = retainPtr(self), capturedBlock = makeBlockPtr(completionHandler)] (Expected<void, Inspector::ExtensionError>&& result) mutable {
        if (!result) {
            capturedBlock([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(result.error())}]);
            return;
        }

        capturedBlock(nil);
    });
#endif
}

- (void)navigateExtensionTabWithIdentifier:(NSString *)extensionTabIdentifier toURL:(NSURL *)url completionHandler:(void(^)(NSError *))completionHandler
{
#if ENABLE(INSPECTOR_EXTENSIONS)
    // It is an error to call this method prior to creating a frontend (i.e., with -connect or -show).
    if (!m_remoteInspectorProxy->extensionController()) {
        completionHandler([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(Inspector::ExtensionError::InvalidRequest) }]);
        return;
    }

    m_remoteInspectorProxy->extensionController()->navigateTabForExtension(extensionTabIdentifier, url, [protectedSelf = retainPtr(self), capturedBlock = makeBlockPtr(completionHandler)] (const std::optional<Inspector::ExtensionError> result) mutable {
        if (result) {
            capturedBlock([NSError errorWithDomain:WKErrorDomain code:WKErrorUnknown userInfo:@{ NSLocalizedFailureReasonErrorKey: Inspector::extensionErrorToString(result.value()) }]);
            return;
        }

        capturedBlock(nil);
    });
#endif
}

@end

#endif // PLATFORM(MAC)
