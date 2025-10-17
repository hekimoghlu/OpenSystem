/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#include "MediaCaptureStatusBarManager.h"

#if ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)

#include "Logging.h"
#include <pal/spi/ios/SBSStatusBarSPI.h>
#include <wtf/BlockPtr.h>
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/TZoneMallocInlines.h>

#include <pal/cocoa/AVFoundationSoftLink.h>

SOFT_LINK_PRIVATE_FRAMEWORK_OPTIONAL(SpringBoardServices)
SOFT_LINK_CLASS_OPTIONAL(SpringBoardServices, SBSStatusBarStyleOverridesAssertion)
SOFT_LINK_CLASS_OPTIONAL(SpringBoardServices, SBSStatusBarStyleOverridesCoordinator)

using namespace WebCore;

@interface WebCoreMediaCaptureStatusBarHandler : NSObject<SBSStatusBarStyleOverridesCoordinatorDelegate>
-(id)initWithManager:(MediaCaptureStatusBarManager*)manager;
-(void)validateIsStopped;
@end

@implementation WebCoreMediaCaptureStatusBarHandler {
    WeakPtr<MediaCaptureStatusBarManager> m_manager;
    RetainPtr<SBSStatusBarStyleOverridesAssertion> m_statusBarStyleOverride;
    RetainPtr<SBSStatusBarStyleOverridesCoordinator> m_coordinator;
}

- (id)initWithManager:(MediaCaptureStatusBarManager*)manager
{
    self = [self init];
    if (!self)
        return nil;

    m_manager = WeakPtr { *manager };
    m_statusBarStyleOverride = nil;
    m_coordinator = nil;
    return self;
}

- (void)validateIsStopped
{
    RELEASE_LOG_ERROR_IF(!!m_statusBarStyleOverride || !!m_coordinator, WebRTC, "WebCoreMediaCaptureStatusBarHandler is not correctly stopped");
    ASSERT(!m_statusBarStyleOverride);
    ASSERT(!m_coordinator);
}

- (void)start
{
    ASSERT(!m_statusBarStyleOverride);
    ASSERT(!m_coordinator);

    UIStatusBarStyleOverrides overrides = UIStatusBarStyleOverrideWebRTCAudioCapture;
    m_statusBarStyleOverride = [getSBSStatusBarStyleOverridesAssertionClass() assertionWithStatusBarStyleOverrides:overrides forPID:legacyPresentingApplicationPID() exclusive:YES showsWhenForeground:YES];
    m_coordinator = adoptNS([[getSBSStatusBarStyleOverridesCoordinatorClass() alloc] init]);
    m_coordinator.get().delegate = self;

    [m_coordinator setRegisteredStyleOverrides:overrides reply:^(NSError *error) {
        if (!error)
            return;
        RELEASE_LOG_ERROR(WebRTC, "WebCoreMediaCaptureStatusBarHandler _acquireStatusBarOverride failed, code = %ld, description is '%s'", [error code], [error localizedDescription].UTF8String);

        callOnMainThread([self, strongSelf = retainPtr(self)] {
            if (m_manager)
                m_manager->didError();
        });
    }];

    // FIXME: Set m_statusBarStyleOverride statusString
    [m_statusBarStyleOverride acquireWithHandler:^(BOOL acquired) {
        if (acquired)
            return;
        callOnMainThread([self, strongSelf = retainPtr(self)] {
            if (m_manager)
                m_manager->didError();
        });
    } invalidationHandler:^{
        callOnMainThread([self, strongSelf = retainPtr(self)] {
            if (m_manager)
                m_manager->didError();
        });
    }];
}

- (void)stop
{
    if (m_coordinator) {
        m_coordinator.get().delegate = nil;
        m_coordinator = nil;
    }

    if (m_statusBarStyleOverride) {
        [m_statusBarStyleOverride invalidate];
        m_statusBarStyleOverride = nil;
    }
    m_manager = nullptr;
}
// FIXME rdar://103273450 (Replace call to SBSStatusBarTapContext with non-deprecated API)
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
- (BOOL)statusBarCoordinator:(SBSStatusBarStyleOverridesCoordinator *)coordinator receivedTapWithContext:(id<SBSStatusBarTapContext>)tapContext completionBlock:(void (^)(void))completion
ALLOW_DEPRECATED_DECLARATIONS_END
{
    callOnMainThread([self, strongSelf = retainPtr(self), completion = makeBlockPtr(completion)]() mutable {
        if (!m_manager)
            return;
        m_manager->didTap([completion = WTFMove(completion)] {
            completion.get()();
        });
    });

    return YES;
}

- (void)statusBarCoordinator:(SBSStatusBarStyleOverridesCoordinator *)coordinator invalidatedRegistrationWithError:(NSError *)error
{
    callOnMainThread([self, strongSelf = retainPtr(self)] {
        if (m_manager)
            m_manager->didError();
    });
}

@end

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaCaptureStatusBarManager);

bool MediaCaptureStatusBarManager::hasSupport()
{
    return SpringBoardServicesLibrary();
}

MediaCaptureStatusBarManager::~MediaCaptureStatusBarManager()
{
    if (m_handler)
        [m_handler validateIsStopped];
}

void MediaCaptureStatusBarManager::start()
{
    m_handler = adoptNS([[WebCoreMediaCaptureStatusBarHandler alloc] initWithManager:this]);
    [m_handler start];
}

void MediaCaptureStatusBarManager::stop()
{
    [m_handler stop];
}

}

#endif // ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)
