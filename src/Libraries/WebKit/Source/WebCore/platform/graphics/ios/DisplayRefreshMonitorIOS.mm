/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#import "DisplayRefreshMonitorIOS.h"

#if PLATFORM(IOS_FAMILY)

#import "DisplayUpdate.h"
#import "Logging.h"
#import "WebCoreThread.h"
#import <QuartzCore/CADisplayLink.h>
#import <wtf/MainThread.h>
#import <wtf/text/TextStream.h>

using WebCore::DisplayRefreshMonitorIOS;

constexpr WebCore::FramesPerSecond DisplayLinkFramesPerSecond = 60;

@interface WebDisplayLinkHandler : NSObject
{
    DisplayRefreshMonitorIOS* m_monitor;
    CADisplayLink *m_displayLink;
}

- (id)initWithMonitor:(DisplayRefreshMonitorIOS*)monitor;
- (void)handleDisplayLink:(CADisplayLink *)sender;
- (void)setPaused:(BOOL)paused;
- (void)invalidate;

@end

@implementation WebDisplayLinkHandler

- (id)initWithMonitor:(DisplayRefreshMonitorIOS*)monitor
{
    if (self = [super init]) {
        m_monitor = monitor;
        // Note that CADisplayLink retains its target (self), so a call to -invalidate is needed on teardown.
        m_displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(handleDisplayLink:)];
        [m_displayLink addToRunLoop:WebThreadNSRunLoop() forMode:NSDefaultRunLoopMode];
        m_displayLink.preferredFramesPerSecond = DisplayLinkFramesPerSecond;
    }
    return self;
}

- (void)dealloc
{
    ASSERT(!m_displayLink); // -invalidate should have been called already.
    [super dealloc];
}

- (void)handleDisplayLink:(CADisplayLink *)sender
{
    UNUSED_PARAM(sender);
    ASSERT(isMainThread());
    
    m_monitor->displayLinkCallbackFired();
}

- (void)setPaused:(BOOL)paused
{
    [m_displayLink setPaused:paused];
}

- (void)invalidate
{
    [m_displayLink invalidate];
    m_displayLink = nullptr;
}

@end

namespace WebCore {

constexpr unsigned maxUnscheduledFireCount { 1 };

DisplayRefreshMonitorIOS::DisplayRefreshMonitorIOS(PlatformDisplayID displayID)
    : DisplayRefreshMonitor(displayID)
{
    setMaxUnscheduledFireCount(maxUnscheduledFireCount);
}

DisplayRefreshMonitorIOS::~DisplayRefreshMonitorIOS()
{
    ASSERT(!m_handler);
}

void DisplayRefreshMonitorIOS::stop()
{
    [m_handler invalidate];
    m_handler = nil;
}

void DisplayRefreshMonitorIOS::displayLinkCallbackFired()
{
    displayLinkFired(m_currentUpdate);
    m_currentUpdate = m_currentUpdate.nextUpdate();
}

bool DisplayRefreshMonitorIOS::startNotificationMechanism()
{
    if (m_displayLinkIsActive)
        return true;

    if (!m_handler) {
        LOG_WITH_STREAM(DisplayLink, stream << "DisplayRefreshMonitorIOS::startNotificationMechanism - creating WebDisplayLinkHandler");
        m_handler = adoptNS([[WebDisplayLinkHandler alloc] initWithMonitor:this]);
    }

    LOG_WITH_STREAM(DisplayLink, stream << "DisplayRefreshMonitorIOS::startNotificationMechanism - starting WebDisplayLinkHandler");
    [m_handler setPaused:NO];

    m_currentUpdate = { 0, DisplayLinkFramesPerSecond };
    m_displayLinkIsActive = true;

    return true;
}

void DisplayRefreshMonitorIOS::stopNotificationMechanism()
{
    if (!m_displayLinkIsActive)
        return;

    LOG_WITH_STREAM(DisplayLink, stream << "DisplayRefreshMonitorIOS::stopNotificationMechanism - pausing WebDisplayLinkHandler");
    [m_handler setPaused:YES];
    m_displayLinkIsActive = false;
}

std::optional<FramesPerSecond> DisplayRefreshMonitorIOS::displayNominalFramesPerSecond()
{
    return DisplayLinkFramesPerSecond;
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
