/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
#import "WKSOAuthorizationDelegate.h"

#if HAVE(APP_SSO)

#import "Logging.h"
#import "SOAuthorizationSession.h"
#import "WebPageProxy.h"
#import <wtf/RunLoop.h>

#define WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG(fmt, ...) RELEASE_LOG(AppSSO, "%p - WKSOAuthorizationDelegate::" fmt, &self, ##__VA_ARGS__)

@implementation WKSOAuthorizationDelegate

- (void)authorization:(SOAuthorization *)authorization presentViewController:(SOAuthorizationViewController)viewController withCompletion:(void (^)(BOOL success, NSError *error))completion
{
    ASSERT(RunLoop::isMain() && completion);
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization (authorization = %p, _session = %p)", authorization, _session.get());
    if (!_session) {
        WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization: No session, so completing with NO as success state.");
        ASSERT_NOT_REACHED();
        completion(NO, nil);
        return;
    }

    if (!viewController) {
        WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization: No view controller to present, so completing with NO as success state.");
        completion(NO, nil);
        return;
    }

    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization: presentingViewController %p", viewController);
    _session->presentViewController(viewController, completion);
}

- (void)authorizationDidNotHandle:(SOAuthorization *)authorization
{
    ASSERT(RunLoop::isMain());
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorizationDidNotHandle: (authorization = %p, _session = %p)", authorization, _session.get());
    LOG_ERROR("Could not handle AppSSO.");
    if (!_session) {
        WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorizationDidNotHandle: No session, so returning early.");
        ASSERT_NOT_REACHED();
        return;
    }
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorizationDidNotHandle: Falling back to web path.");
    _session->fallBackToWebPath();
}

- (void)authorizationDidCancel:(SOAuthorization *)authorization
{
    ASSERT(RunLoop::isMain());
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorizationDidCancel: (authorization = %p, _session = %p)", authorization, _session.get());
    if (!_session) {
        WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorizationDidCancel: No session, so returning early.");
        ASSERT_NOT_REACHED();
        return;
    }
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorizationDidCancel: Aborting session.");
    _session->abort();
}

- (void)authorizationDidComplete:(SOAuthorization *)authorization
{
    ASSERT(RunLoop::isMain());
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorizationDidComplete: (authorization = %p, _session = %p)", authorization, _session.get());
    LOG_ERROR("Complete AppSSO without any data.");
    if (!_session) {
        WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorizationDidComplete: No session, so returning early.");
        ASSERT_NOT_REACHED();
        return;
    }
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorizationDidComplete: Falling back to web path.");
    _session->fallBackToWebPath();
}

- (void)authorization:(SOAuthorization *)authorization didCompleteWithHTTPAuthorizationHeaders:(NSDictionary *)httpAuthorizationHeaders
{
    ASSERT(RunLoop::isMain());
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization:didCompleteWithHTTPAuthorizationHeaders: (authorization = %p, _session = %p)", authorization, _session.get());
    LOG_ERROR("Complete AppSSO with unexpected callback.");
    if (!_session) {
        WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization:didCompleteWithHTTPAuthorizationHeaders: No session, so returning early.");
        ASSERT_NOT_REACHED();
        return;
    }
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization:didCompleteWithHTTPAuthorizationHeaders: Falling back to web path.");
    _session->fallBackToWebPath();
}

- (void)authorization:(SOAuthorization *)authorization didCompleteWithHTTPResponse:(NSHTTPURLResponse *)httpResponse httpBody:(NSData *)httpBody
{
    ASSERT(RunLoop::isMain());
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization:didCompleteWithHTTPResponse: (authorization = %p, _session = %p)", authorization, _session.get());
    if (!_session) {
        WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization:didCompleteWithHTTPResponse: No session, so returning early.");
        ASSERT_NOT_REACHED();
        return;
    }
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization:didCompleteWithHTTPResponse: Completing.");
    _session->complete(httpResponse, httpBody);
}

- (void)authorization:(SOAuthorization *)authorization didCompleteWithError:(NSError *)error
{
    ASSERT(RunLoop::isMain());
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization:didCompleteWithError: (authorization = %p, _session = %p)", authorization, _session.get());
    if (error.code)
        LOG_ERROR("Could not complete AppSSO operation. Error: %d", error.code);
    if (!_session) {
        WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization:didCompleteWithError: No session, so returning early.");
        ASSERT_NOT_REACHED();
        return;
    }
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("authorization:didCompleteWithError: Falling back to web path.");
    _session->fallBackToWebPath();
}

- (void)setSession:(RefPtr<WebKit::SOAuthorizationSession>&&)session
{
    RELEASE_ASSERT(RunLoop::isMain());
    WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG("setSession: (existing session = %p, new session = %p)", _session.get(), session.get());
    _session = WTFMove(session);

    if (_session)
        _session->shouldStart();
}
@end

#undef WKSOAUTHORIZATIONDELEGATE_RELEASE_LOG

#endif
