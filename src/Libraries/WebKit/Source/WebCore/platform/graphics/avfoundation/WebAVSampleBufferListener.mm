/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#import "WebAVSampleBufferListener.h"

#import "WebSampleBufferVideoRendering.h"
#import <Foundation/Foundation.h>
#import <Foundation/NSObject.h>
#import <pal/avfoundation/MediaTimeAVFoundation.h>
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <wtf/RetainPtr.h>
#import <wtf/ThreadSafetyAnalysis.h>
#import <wtf/Vector.h>

#pragma mark - Soft Linking

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

#if HAVE(AVSAMPLEBUFFERDISPLAYLAYER_READYFORDISPLAY)
// FIXME (117934497): Remove staging code once -[AVSampleBufferDisplayLayer isReadyForDisplay] is available in SDKs used by WebKit builders
@interface AVSampleBufferDisplayLayer (Staging_113656776)
@property (nonatomic, readonly, getter=isReadyForDisplay) BOOL readyForDisplay;
@end
#endif

static NSString * const errorKeyPath = @"error";
static NSString * const outputObscuredDueToInsufficientExternalProtectionKeyPath = @"outputObscuredDueToInsufficientExternalProtection";

static void* errorContext = &errorContext;
static void* outputObscuredDueToInsufficientExternalProtectionContext = &outputObscuredDueToInsufficientExternalProtectionContext;

namespace WebCore {
static bool isSampleBufferVideoRenderer(id object)
{
    if (is_objc<AVSampleBufferDisplayLayer>(object))
        return true;

#if HAVE(AVSAMPLEBUFFERVIDEORENDERER)
    if (is_objc<AVSampleBufferVideoRenderer>(object))
        return true;
#endif

    return false;
}
}

@interface WebAVSampleBufferListenerPrivate : NSObject {
    WeakPtr<WebCore::WebAVSampleBufferListenerClient> _client WTF_GUARDED_BY_CAPABILITY(mainThread);
    Vector<RetainPtr<WebSampleBufferVideoRendering>> _videoRenderers;
    Vector<RetainPtr<AVSampleBufferAudioRenderer>> _audioRenderers;
}

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithClient:(WebCore::WebAVSampleBufferListenerClient &)client NS_DESIGNATED_INITIALIZER;
- (void)invalidate;
- (void)beginObservingVideoRenderer:(WebSampleBufferVideoRendering *)renderer;
- (void)stopObservingVideoRenderer:(WebSampleBufferVideoRendering *)renderer;
- (void)beginObservingAudioRenderer:(AVSampleBufferAudioRenderer *)renderer;
- (void)stopObservingAudioRenderer:(AVSampleBufferAudioRenderer *)renderer;

@end

@implementation WebAVSampleBufferListenerPrivate

- (instancetype)initWithClient:(WebCore::WebAVSampleBufferListenerClient &)client
{
    assertIsMainThread();
    if (!(self = [super init]))
        return nil;

    _client = client;
    return self;
}

- (void)invalidate
{
    assertIsMainThread();

    _client = nullptr;

    for (auto& renderer : _videoRenderers) {
        [renderer removeObserver:self forKeyPath:errorKeyPath];
        [renderer removeObserver:self forKeyPath:outputObscuredDueToInsufficientExternalProtectionKeyPath];
    }
    _videoRenderers.clear();

    for (auto& renderer : _audioRenderers)
        [renderer removeObserver:self forKeyPath:errorKeyPath];
    _audioRenderers.clear();

    [NSNotificationCenter.defaultCenter removeObserver:self];
}

- (void)beginObservingVideoRenderer:(WebSampleBufferVideoRendering *)renderer
{
    ASSERT(isMainThread());
    ASSERT(!_videoRenderers.contains(renderer));

    _videoRenderers.append(renderer);
    [renderer addObserver:self forKeyPath:errorKeyPath options:NSKeyValueObservingOptionNew context:errorContext];
    [renderer addObserver:self forKeyPath:outputObscuredDueToInsufficientExternalProtectionKeyPath options:NSKeyValueObservingOptionNew context:outputObscuredDueToInsufficientExternalProtectionContext];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(layerFailedToDecode:) name:AVSampleBufferDisplayLayerFailedToDecodeNotification object:renderer];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(layerFailedToDecode:) name:AVSampleBufferVideoRendererDidFailToDecodeNotification object:renderer];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(layerRequiresFlushToResumeDecodingChanged:) name:AVSampleBufferDisplayLayerRequiresFlushToResumeDecodingDidChangeNotification object:renderer];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(layerRequiresFlushToResumeDecodingChanged:) name:AVSampleBufferVideoRendererRequiresFlushToResumeDecodingDidChangeNotification object:renderer];

#if HAVE(AVSAMPLEBUFFERDISPLAYLAYER_READYFORDISPLAY)
    // FIXME (117934497): Remove staging code once -[AVSampleBufferDisplayLayer isReadyForDisplay] is available in SDKs used by WebKit builders
    if (PAL::canLoad_AVFoundation_AVSampleBufferDisplayLayerReadyForDisplayDidChangeNotification())
        [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(layerReadyForDisplayChanged:) name:AVSampleBufferDisplayLayerReadyForDisplayDidChangeNotification object:renderer];
#endif
}

- (void)stopObservingVideoRenderer:(WebSampleBufferVideoRendering *)renderer
{
    ASSERT(isMainThread());
    ASSERT(_videoRenderers.contains(renderer));

    [renderer removeObserver:self forKeyPath:errorKeyPath];
    [renderer removeObserver:self forKeyPath:outputObscuredDueToInsufficientExternalProtectionKeyPath];
    _videoRenderers.remove(_videoRenderers.find(renderer));

    [NSNotificationCenter.defaultCenter removeObserver:self name:AVSampleBufferDisplayLayerFailedToDecodeNotification object:renderer];
    [NSNotificationCenter.defaultCenter removeObserver:self name:AVSampleBufferVideoRendererDidFailToDecodeNotification object:renderer];
    [NSNotificationCenter.defaultCenter removeObserver:self name:AVSampleBufferDisplayLayerRequiresFlushToResumeDecodingDidChangeNotification object:renderer];
    [NSNotificationCenter.defaultCenter removeObserver:self name:AVSampleBufferVideoRendererRequiresFlushToResumeDecodingDidChangeNotification object:renderer];
#if HAVE(AVSAMPLEBUFFERDISPLAYLAYER_READYFORDISPLAY)
    // FIXME (117934497): Remove staging code once -[AVSampleBufferDisplayLayer isReadyForDisplay] is available in SDKs used by WebKit builders
    if (PAL::canLoad_AVFoundation_AVSampleBufferDisplayLayerReadyForDisplayDidChangeNotification())
        [NSNotificationCenter.defaultCenter removeObserver:self name:AVSampleBufferDisplayLayerReadyForDisplayDidChangeNotification object:renderer];
#endif
}

- (void)beginObservingAudioRenderer:(AVSampleBufferAudioRenderer *)renderer
{
    ASSERT(isMainThread());
    ASSERT(!_audioRenderers.contains(renderer));

    _audioRenderers.append(renderer);
    [renderer addObserver:self forKeyPath:errorKeyPath options:NSKeyValueObservingOptionNew context:errorContext];
    [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(audioRendererWasAutomaticallyFlushed:) name:AVSampleBufferAudioRendererWasFlushedAutomaticallyNotification object:renderer];
}

- (void)stopObservingAudioRenderer:(AVSampleBufferAudioRenderer *)renderer
{
    ASSERT(isMainThread());
    ASSERT(_audioRenderers.contains(renderer));

    [renderer removeObserver:self forKeyPath:errorKeyPath];
    [NSNotificationCenter.defaultCenter removeObserver:self name:AVSampleBufferAudioRendererWasFlushedAutomaticallyNotification object:renderer];

    _audioRenderers.remove(_audioRenderers.find(renderer));
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void*)context
{
    if (context == errorContext) {
        RetainPtr error = dynamic_objc_cast<NSError>([change valueForKey:NSKeyValueChangeNewKey]);
        if (!error)
            return;

        if (WebCore::isSampleBufferVideoRenderer(object)) {
            RetainPtr renderer = (WebSampleBufferVideoRendering *)object;

            ensureOnMainThread([self, protectedSelf = RetainPtr { self }, renderer = WTFMove(renderer), error = WTFMove(error)] {
                ASSERT(_videoRenderers.contains(renderer.get()));
                if (auto client = _client.get())
                    client->videoRendererDidReceiveError(renderer.get(), error.get());
            });
            return;
        }

        if ([object isKindOfClass:PAL::getAVSampleBufferAudioRendererClass()]) {
            RetainPtr renderer = (AVSampleBufferAudioRenderer *)object;

            ensureOnMainThread([self, protectedSelf = RetainPtr { self }, renderer = WTFMove(renderer), error = WTFMove(error)] {
                ASSERT(_audioRenderers.contains(renderer.get()));
                if (auto client = _client.get())
                    client->audioRendererDidReceiveError(renderer.get(), error.get());
            });
            return;
        }

        ASSERT_NOT_REACHED_WITH_MESSAGE("Unexpected kind of object while observing errorContext");
        return;
    }

    if (context == outputObscuredDueToInsufficientExternalProtectionContext) {
        RetainPtr renderer = WebCore::isSampleBufferVideoRenderer(object) ? (WebSampleBufferVideoRendering *)object : nil;
        BOOL isObscured = [[change valueForKey:NSKeyValueChangeNewKey] boolValue];

        ensureOnMainThread([self, protectedSelf = RetainPtr { self }, renderer = WTFMove(renderer), isObscured] {
            ASSERT(_videoRenderers.contains(renderer.get()));
            if (auto client = _client.get())
                client->outputObscuredDueToInsufficientExternalProtectionChanged(isObscured);
        });
        return;
    }

    [super observeValueForKeyPath:keyPath ofObject:object change:change context:context];
}

- (void)layerFailedToDecode:(NSNotification *)notification
{
    RetainPtr renderer = WebCore::isSampleBufferVideoRenderer(notification.object) ? (WebSampleBufferVideoRendering *)notification.object : nil;
    RetainPtr error = dynamic_objc_cast<NSError>([notification.userInfo valueForKey:AVSampleBufferDisplayLayerFailedToDecodeNotificationErrorKey]);

    ensureOnMainThread([self, protectedSelf = RetainPtr { self }, renderer = WTFMove(renderer), error = WTFMove(error)] {
        if (!_videoRenderers.contains(renderer.get()))
            return;
        if (auto client = _client.get())
            client->videoRendererDidReceiveError(renderer.get(), error.get());
    });
}

- (void)layerRequiresFlushToResumeDecodingChanged:(NSNotification *)notification
{
    RetainPtr renderer = WebCore::isSampleBufferVideoRenderer(notification.object) ? (WebSampleBufferVideoRendering *)notification.object : nil;
    BOOL requiresFlush = [renderer requiresFlushToResumeDecoding];

    ensureOnMainThread([self, protectedSelf = RetainPtr { self }, renderer = WTFMove(renderer), requiresFlush] {
        if (!_videoRenderers.contains(renderer.get()))
            return;
        if (auto client = _client.get())
            client->videoRendererRequiresFlushToResumeDecodingChanged(renderer.get(), requiresFlush);
    });
}

#if HAVE(AVSAMPLEBUFFERDISPLAYLAYER_READYFORDISPLAY)
- (void)layerReadyForDisplayChanged:(NSNotification *)notification
{
    RetainPtr layer = dynamic_objc_cast<AVSampleBufferDisplayLayer>(notification.object);
    if (!layer)
        return;

    BOOL isReadyForDisplay = [layer isReadyForDisplay];

    ensureOnMainThread([self, protectedSelf = RetainPtr { self }, layer = WTFMove(layer), isReadyForDisplay] {
        if (!_videoRenderers.contains(layer.get()))
            return;
        if (auto client = _client.get())
            client->videoRendererReadyForDisplayChanged(layer.get(), isReadyForDisplay);
    });
}
#endif

- (void)audioRendererWasAutomaticallyFlushed:(NSNotification *)notification
{
    RetainPtr renderer = dynamic_objc_cast<AVSampleBufferAudioRenderer>(notification.object);
    CMTime flushTime = [[notification.userInfo valueForKey:AVSampleBufferAudioRendererFlushTimeKey] CMTimeValue];

    ensureOnMainThread([self, protectedSelf = RetainPtr { self }, renderer = WTFMove(renderer), flushTime] {
        if (!_audioRenderers.contains(renderer.get()))
            return;
        if (auto client = _client.get())
            client->audioRendererWasAutomaticallyFlushed(renderer.get(), flushTime);
    });
}

@end

namespace WebCore {

WebAVSampleBufferListener::WebAVSampleBufferListener(WebAVSampleBufferListenerClient& client)
    : m_private(adoptNS([[WebAVSampleBufferListenerPrivate alloc] initWithClient:client]))
{
}

void WebAVSampleBufferListener::invalidate()
{
    [m_private invalidate];
}

void WebAVSampleBufferListener::beginObservingVideoRenderer(WebSampleBufferVideoRendering *renderer)
{
    [m_private beginObservingVideoRenderer:renderer];
}

void WebAVSampleBufferListener::stopObservingVideoRenderer(WebSampleBufferVideoRendering *renderer)
{
    [m_private stopObservingVideoRenderer:renderer];
}

void WebAVSampleBufferListener::beginObservingAudioRenderer(AVSampleBufferAudioRenderer* renderer)
{
    [m_private beginObservingAudioRenderer:renderer];
}

void WebAVSampleBufferListener::stopObservingAudioRenderer(AVSampleBufferAudioRenderer* renderer)
{
    [m_private stopObservingAudioRenderer:renderer];
}

} // namespace WebCore
