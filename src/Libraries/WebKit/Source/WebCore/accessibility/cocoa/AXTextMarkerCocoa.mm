/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#import "AXTextMarker.h"

#import <Foundation/NSRange.h>
#import <wtf/StdLibExtras.h>

#if PLATFORM(MAC)
#import "AXIsolatedObject.h"
#import "WebAccessibilityObjectWrapperMac.h"
#import <pal/spi/mac/HIServicesSPI.h>
#else // PLATFORM(IOS_FAMILY)
#import "WebAccessibilityObjectWrapperIOS.h"
#endif

namespace WebCore {

using namespace Accessibility;

AXTextMarker::AXTextMarker(PlatformTextMarkerData platformData)
{
    if (!platformData)
        return;

#if PLATFORM(MAC)
    if (CFGetTypeID(platformData) != AXTextMarkerGetTypeID()) {
        ASSERT_NOT_REACHED();
        return;
    }

    if (AXTextMarkerGetLength(platformData) != sizeof(m_data)) {
        ASSERT_NOT_REACHED();
        return;
    }

    memcpySpan(asMutableByteSpan(m_data), AXTextMarkerGetByteSpan(platformData));
#else // PLATFORM(IOS_FAMILY)
    [platformData getBytes:&m_data length:sizeof(m_data)];
#endif
}

RetainPtr<PlatformTextMarkerData> AXTextMarker::platformData() const
{
#if PLATFORM(MAC)
    return adoptCF(AXTextMarkerCreate(kCFAllocatorDefault, (const UInt8*)&m_data, sizeof(m_data)));
#else // PLATFORM(IOS_FAMILY)
    return [NSData dataWithBytes:&m_data length:sizeof(m_data)];
#endif
}

#if ENABLE(AX_THREAD_TEXT_APIS)
// FIXME: There's a lot of duplicated code between this function and AXTextMarkerRange::toString().
RetainPtr<NSAttributedString> AXTextMarkerRange::toAttributedString(AXCoreObject::SpellCheck spellCheck) const
{
    RELEASE_ASSERT(!isMainThread());

    auto start = m_start.toTextRunMarker();
    if (!start.isValid())
        return nil;
    auto end = m_end.toTextRunMarker();
    if (!end.isValid())
        return nil;

    if (start.isolatedObject() == end.isolatedObject()) {
        size_t minOffset = std::min(start.offset(), end.offset());
        size_t maxOffset = std::max(start.offset(), end.offset());
        // FIXME: createAttributedString takes a StringView, but we create a full-fledged String. Could we create a
        // new substringView method that returns a StringView?
        return start.isolatedObject()->createAttributedString(start.runs()->substring(minOffset, maxOffset - minOffset), spellCheck).autorelease();
    }

    RetainPtr<NSMutableAttributedString> result = start.isolatedObject()->createAttributedString(start.runs()->substring(start.offset()), spellCheck);
    auto emitNewlineOnExit = [&] (AXIsolatedObject& object) {
        // FIXME: This function should not just be emitting newlines, but instead handling every character type in TextEmissionBehavior.
        auto behavior = object.emitTextAfterBehavior();
        if (behavior != TextEmissionBehavior::Newline && behavior != TextEmissionBehavior::DoubleNewline)
            return;

        auto length = [result length];
        // Like TextIterator, don't emit a newline if the most recently emitted character was already a newline.
        if (length && [[result string] characterAtIndex:length - 1] != '\n') {
            // FIXME: This is super inefficient. We are creating a whole new dictionary and attributed string just to append newline(s).
            NSString *newlineString = behavior == TextEmissionBehavior::Newline ? @"\n" : @"\n\n";
            NSDictionary *attributes = [result attributesAtIndex:length - 1 effectiveRange:nil];
            [result appendAttributedString:adoptNS([[NSAttributedString alloc] initWithString:newlineString attributes:attributes]).get()];
        }
    };

    // FIXME: If we've been given reversed markers, i.e. the end marker actually comes before the start marker,
    // we may want to detect this and try searching AXDirection::Previous?
    RefPtr current = findObjectWithRuns(*start.isolatedObject(), AXDirection::Next, std::nullopt, emitNewlineOnExit);
    while (current && current->objectID() != end.objectID()) {
        [result appendAttributedString:current->createAttributedString(current->textRuns()->toString(), spellCheck).autorelease()];
        current = findObjectWithRuns(*current, AXDirection::Next, std::nullopt, emitNewlineOnExit);
    }
    [result appendAttributedString:end.isolatedObject()->createAttributedString(end.runs()->substring(0, end.offset()), spellCheck).autorelease()];

    return result;
}
#endif // ENABLE(AX_THREAD_TEXT_APIS)

#if PLATFORM(MAC)

AXTextMarkerRange::AXTextMarkerRange(AXTextMarkerRangeRef textMarkerRangeRef)
{
    if (!textMarkerRangeRef || CFGetTypeID(textMarkerRangeRef) != AXTextMarkerRangeGetTypeID()) {
        ASSERT_NOT_REACHED();
        return;
    }

    auto start = AXTextMarkerRangeCopyStartMarker(textMarkerRangeRef);
    auto end = AXTextMarkerRangeCopyEndMarker(textMarkerRangeRef);

    m_start = start;
    m_end = end;

    CFRelease(start);
    CFRelease(end);
}

RetainPtr<AXTextMarkerRangeRef> AXTextMarkerRange::platformData() const
{
    return adoptCF(AXTextMarkerRangeCreate(kCFAllocatorDefault
        , m_start.platformData().autorelease()
        , m_end.platformData().autorelease()
    ));
}

#elif PLATFORM(IOS_FAMILY)

AXTextMarkerRange::AXTextMarkerRange(NSArray *markers)
{
    if (markers.count != 2)
        return;

    WebAccessibilityTextMarker *start = [markers objectAtIndex:0];
    WebAccessibilityTextMarker *end = [markers objectAtIndex:1];
    if (![start isKindOfClass:[WebAccessibilityTextMarker class]] || ![end isKindOfClass:[WebAccessibilityTextMarker class]])
        return;

    m_start = { [start textMarkerData ] };
    m_end = { [end textMarkerData] };
}

RetainPtr<NSArray> AXTextMarkerRange::platformData() const
{
    if (!*this)
        return nil;

    RefPtr object = m_start.object();
    ASSERT(object); // Since *this is not null.
    auto* cache = object->axObjectCache();
    if (!cache)
        return nil;

    auto start = adoptNS([[WebAccessibilityTextMarker alloc] initWithTextMarker:&m_start.m_data cache:cache]);
    auto end = adoptNS([[WebAccessibilityTextMarker alloc] initWithTextMarker:&m_end.m_data cache:cache]);
    return adoptNS([[NSArray alloc] initWithObjects:start.get(), end.get(), nil]);
}

#endif // PLATFORM(IOS_FAMILY)

std::optional<NSRange> AXTextMarkerRange::nsRange() const
{
    return characterRange();
}

} // namespace WebCore
