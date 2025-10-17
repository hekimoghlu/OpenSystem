/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
#import "GroupActivitiesCoordinator.h"

#if ENABLE(MEDIA_SESSION_COORDINATOR) && HAVE(GROUP_ACTIVITIES)

#import "WKGroupSession.h"
#import <WebCore/NotImplemented.h>
#import <wtf/BlockPtr.h>
#import <wtf/TZoneMallocInlines.h>

#import <pal/cocoa/AVFoundationSoftLink.h>
#import <pal/cf/CoreMediaSoftLink.h>

@interface WKGroupActivitiesCoordinatorDelegate : NSObject<AVPlaybackCoordinatorPlaybackControlDelegate> {
    WeakPtr<WebKit::GroupActivitiesCoordinator> _parent;
}
- (id)initWithParent:(WebKit::GroupActivitiesCoordinator&)parent;
@end

@implementation WKGroupActivitiesCoordinatorDelegate
- (id)initWithParent:(WebKit::GroupActivitiesCoordinator&)parent
{
    self = [super init];
    if (!self)
        return nil;

    _parent = parent;
    return self;
}

-(void)playbackCoordinator:(AVDelegatingPlaybackCoordinator *)coordinator didIssuePlayCommand:(AVDelegatingPlaybackCoordinatorPlayCommand *)playCommand completionHandler:(void (^)(void))completionHandler {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (_parent) {
            _parent->issuePlayCommand(playCommand, [completionHandler = makeBlockPtr(completionHandler)] {
                completionHandler();
            });
        }
    });
}

-(void)playbackCoordinator:(AVDelegatingPlaybackCoordinator *)coordinator didIssuePauseCommand:(AVDelegatingPlaybackCoordinatorPauseCommand *)pauseCommand completionHandler:(void (^)(void))completionHandler {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (!_parent)
            return;
        _parent->issuePauseCommand(pauseCommand, [completionHandler = makeBlockPtr(completionHandler)] {
            completionHandler();
        });
    });
}

-(void)playbackCoordinator:(AVDelegatingPlaybackCoordinator *)coordinator didIssueSeekCommand:(AVDelegatingPlaybackCoordinatorSeekCommand *)seekCommand completionHandler:(void (^)(void))completionHandler {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (!_parent)
            return;
        _parent->issueSeekCommand(seekCommand, [completionHandler = makeBlockPtr(completionHandler)] {
            completionHandler();
        });
    });
}

-(void)playbackCoordinator:(AVDelegatingPlaybackCoordinator *)coordinator didIssueBufferingCommand:(AVDelegatingPlaybackCoordinatorBufferingCommand *)bufferingCommand completionHandler:(void (^)(void))completionHandler {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (!_parent)
            return;
        _parent->issueBufferingCommand(bufferingCommand, [completionHandler = makeBlockPtr(completionHandler)] {
            completionHandler();
        });
    });
}

-(void)playbackCoordinator:(AVDelegatingPlaybackCoordinator *)coordinator didIssuePrepareTransitionCommand:(AVDelegatingPlaybackCoordinatorPrepareTransitionCommand *)prepareTransitionCommand {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (_parent)
            _parent->issuePrepareTransitionCommand(prepareTransitionCommand);
    });
}
@end

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(GroupActivitiesCoordinator);

Ref<GroupActivitiesCoordinator> GroupActivitiesCoordinator::create(GroupActivitiesSession& session)
{
    return adoptRef(*new GroupActivitiesCoordinator(session));
}

GroupActivitiesCoordinator::GroupActivitiesCoordinator(GroupActivitiesSession& session)
    : m_session(session)
    , m_delegate(adoptNS([[WKGroupActivitiesCoordinatorDelegate alloc] initWithParent:*this]))
    , m_playbackCoordinator(adoptNS([PAL::allocAVDelegatingPlaybackCoordinatorInstance() initWithPlaybackControlDelegate:m_delegate.get()]))
    , m_stateChangeObserver([this] (auto& session, auto state) { sessionStateChanged(session, state); })
{
    [session.groupSession() coordinateWithCoordinator:m_playbackCoordinator.get()];
    session.addStateChangeObserver(m_stateChangeObserver);
}

GroupActivitiesCoordinator::~GroupActivitiesCoordinator()
{
    m_session->groupSession().newActivityCallback = nil;
    m_session->groupSession().stateChangedCallback = nil;
}

void GroupActivitiesCoordinator::sessionStateChanged(const GroupActivitiesSession& session, GroupActivitiesSession::State state)
{
    if (!client())
        return;

    static_assert(static_cast<size_t>(MediaSessionCoordinatorState::Waiting) == static_cast<size_t>(GroupActivitiesSession::State::Waiting), "MediaSessionCoordinatorState::Waiting != WKGroupSessionStateWaiting");
    static_assert(static_cast<size_t>(MediaSessionCoordinatorState::Joined) == static_cast<size_t>(GroupActivitiesSession::State::Joined), "MediaSessionCoordinatorState::Joined != WKGroupSessionStateJoined");
    static_assert(static_cast<size_t>(MediaSessionCoordinatorState::Closed) == static_cast<size_t>(GroupActivitiesSession::State::Invalidated), "MediaSessionCoordinatorState::Closed != WKGroupSessionStateInvalidated");
    client()->coordinatorStateChanged(static_cast<MediaSessionCoordinatorState>(state));
}

String GroupActivitiesCoordinator::identifier() const
{
    return m_session->uuid();
}

void GroupActivitiesCoordinator::join(CoordinatorCompletionHandler&& callback)
{
    m_session->join();
    callback(std::nullopt);
}

void GroupActivitiesCoordinator::leave()
{
    m_session->leave();
}

void GroupActivitiesCoordinator::seekTo(double time, CoordinatorCompletionHandler&& callback)
{
    [m_playbackCoordinator coordinateSeekToTime:PAL::CMTimeMakeWithSeconds(time, 1000) options:0];
    callback(std::nullopt);
}

void GroupActivitiesCoordinator::play(CoordinatorCompletionHandler&& callback)
{
    [m_playbackCoordinator coordinateRateChangeToRate:1 options:0];
    callback(std::nullopt);
}

void GroupActivitiesCoordinator::pause(CoordinatorCompletionHandler&& callback)
{
    [m_playbackCoordinator coordinateRateChangeToRate:0 options:0];
    callback(std::nullopt);
}

void GroupActivitiesCoordinator::setTrack(const String& track, CoordinatorCompletionHandler&& callback)
{
    callback(std::nullopt);
}

void GroupActivitiesCoordinator::positionStateChanged(const std::optional<MediaPositionState>& positionState)
{
    if (m_positionState == positionState)
        return;
    m_positionState = positionState;
}

void GroupActivitiesCoordinator::readyStateChanged(MediaSessionReadyState readyState)
{
    if (m_readyState == readyState)
        return;
    m_readyState = readyState;
}

void GroupActivitiesCoordinator::playbackStateChanged(MediaSessionPlaybackState playbackState)
{
    if (m_playbackState == playbackState)
        return;
    m_playbackState = playbackState;
}

void GroupActivitiesCoordinator::trackIdentifierChanged(const String& identifier)
{
    if (identifier != String([m_playbackCoordinator currentItemIdentifier]))
        [m_playbackCoordinator transitionToItemWithIdentifier:identifier proposingInitialTimingBasedOnTimebase:nil];
}

void GroupActivitiesCoordinator::issuePlayCommand(AVDelegatingPlaybackCoordinatorPlayCommand *playCommand, CommandCompletionHandler&& callback)
{
    if (!client()) {
        callback();
        return;
    }

    std::optional<double> itemTime;
    if (CMTIME_IS_NUMERIC(playCommand.itemTime))
        itemTime = PAL::CMTimeGetSeconds(playCommand.itemTime);
    std::optional<MonotonicTime> hostTime;
    if (CMTIME_IS_NUMERIC(playCommand.hostClockTime))
        hostTime = MonotonicTime::fromMachAbsoluteTime(PAL::CMClockConvertHostTimeToSystemUnits(playCommand.hostClockTime));

    client()->playSession(itemTime, hostTime, [callback = WTFMove(callback)] (bool) {
        callback();
    });
}

void GroupActivitiesCoordinator::issuePauseCommand(AVDelegatingPlaybackCoordinatorPauseCommand *pauseCommand, CommandCompletionHandler&& callback)
{
    if (!client()) {
        callback();
        return;
    }

    client()->pauseSession([callback = WTFMove(callback)] (bool) {
        callback();
    });
}

void GroupActivitiesCoordinator::issueSeekCommand(AVDelegatingPlaybackCoordinatorSeekCommand *seekCommand, CommandCompletionHandler&& callback)
{
    if (!client()) {
        callback();
        return;
    }

    if (!CMTIME_IS_NUMERIC(seekCommand.itemTime)) {
        ASSERT_NOT_REACHED();
        callback();
        return;
    }

    client()->seekSessionToTime(PAL::CMTimeGetSeconds(seekCommand.itemTime), [callback = WTFMove(callback)] (bool) mutable {
        callback();
    });
}

void GroupActivitiesCoordinator::issueBufferingCommand(AVDelegatingPlaybackCoordinatorBufferingCommand *, CommandCompletionHandler&& completionHandler)
{
    completionHandler();
    notImplemented();
}

void GroupActivitiesCoordinator::issuePrepareTransitionCommand(AVDelegatingPlaybackCoordinatorPrepareTransitionCommand *)
{
    notImplemented();
}

}

#endif
