/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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

#if (PLATFORM(IOS_FAMILY) || (PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE))) && ENABLE(VIDEO)
#import "TextTrackRepresentationCocoa.h"

#import "FloatRect.h"
#import "GraphicsContextCG.h"
#import "IntRect.h"

#if PLATFORM(IOS_FAMILY)
#import "WebCoreThread.h"
#import "WebCoreThreadRun.h"
#endif

#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/TZoneMallocInlines.h>


@interface WebCoreTextTrackRepresentationCocoaHelper : NSObject <CALayerDelegate> {
    WebCore::TextTrackRepresentationCocoa* _parent;
}
- (id)initWithParent:(WebCore::TextTrackRepresentationCocoa*)parent;
@property (assign) WebCore::TextTrackRepresentationCocoa* parent;
@end

@implementation WebCoreTextTrackRepresentationCocoaHelper
- (id)initWithParent:(WebCore::TextTrackRepresentationCocoa*)parent
{
    if (!(self = [super init]))
        return nil;

    self.parent = parent;

    return self;
}

- (void)dealloc
{
    self.parent = nullptr;
    [super dealloc];
}

- (void)setParent:(WebCore::TextTrackRepresentationCocoa*)parent
{
    if (_parent)
        [_parent->platformLayer() removeObserver:self forKeyPath:@"bounds"];

    _parent = parent;

    if (_parent)
        [_parent->platformLayer() addObserver:self forKeyPath:@"bounds" options:0 context:0];
}

- (WebCore::TextTrackRepresentationCocoa*)parent
{
    return _parent;
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context
{
    UNUSED_PARAM(change);
    UNUSED_PARAM(context);
#if PLATFORM(IOS_FAMILY)
    WebThreadRun(^{
        if (_parent && [keyPath isEqual:@"bounds"] && object == _parent->platformLayer())
            _parent->client().textTrackRepresentationBoundsChanged(_parent->bounds());
    });
#else
    if (_parent && [keyPath isEqual:@"bounds"] && object == _parent->platformLayer())
        _parent->boundsChanged();
#endif
}

- (id)actionForLayer:(CALayer *)layer forKey:(NSString *)event
{
    UNUSED_PARAM(layer);
    UNUSED_PARAM(event);
    // Returning a NSNull from this delegate method disables all implicit CALayer actions.
    return [NSNull null];
}

@end

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextTrackRepresentationCocoa);

std::unique_ptr<TextTrackRepresentation> TextTrackRepresentation::create(TextTrackRepresentationClient& client, HTMLMediaElement& mediaElement)
{
    if (TextTrackRepresentationCocoa::representationFactory())
        return TextTrackRepresentationCocoa::representationFactory()(client, mediaElement);
    return makeUnique<TextTrackRepresentationCocoa>(client);
}

TextTrackRepresentationCocoa::TextTrackRepresentationFactory& TextTrackRepresentationCocoa::representationFactory()
{
    static NeverDestroyed<TextTrackRepresentationFactory> factory;
    return factory.get();
}

TextTrackRepresentationCocoa::TextTrackRepresentationCocoa(TextTrackRepresentationClient& client)
    : m_client(client)
    , m_layer(adoptNS([[CALayer alloc] init]))
    , m_delegate(adoptNS([[WebCoreTextTrackRepresentationCocoaHelper alloc] initWithParent:this]))
{
    [m_layer setDelegate:m_delegate.get()];
    [m_layer setContentsGravity:kCAGravityBottom];

    [m_layer setName:@"TextTrackRepresentation"];
}

TextTrackRepresentationCocoa::~TextTrackRepresentationCocoa()
{
    [m_layer setDelegate:nil];
    [m_delegate setParent:nullptr];
}

void TextTrackRepresentationCocoa::update()
{
    if (auto representation = m_client.createTextTrackRepresentationImage())
        [m_layer setContents:(__bridge id)representation->platformImage().get()];
}

void TextTrackRepresentationCocoa::setContentScale(float scale)
{
    [m_layer setContentsScale:scale];
}

void TextTrackRepresentationCocoa::setHidden(bool hidden) const
{
    [m_layer setHidden:hidden];
}

void TextTrackRepresentationCocoa::setBounds(const IntRect& bounds)
{
    [m_layer setBounds:FloatRect(bounds)];
}

IntRect TextTrackRepresentationCocoa::bounds() const
{
    return enclosingIntRect(FloatRect([m_layer bounds]));
}

void TextTrackRepresentationCocoa::boundsChanged()
{
    callOnMainThread([weakThis = WeakPtr { *this }] {
        if (weakThis)
            weakThis->client().textTrackRepresentationBoundsChanged(weakThis->bounds());
    });
}

} // namespace WebCore

#endif // (PLATFORM(IOS_FAMILY) || (PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE))) && ENABLE(VIDEO)
