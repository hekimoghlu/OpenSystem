/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
#import "NSURLUtilities.h"

#if HAVE(NSURL_TITLE)

#import <pal/cocoa/LinkPresentationSoftLink.h>

#if !HAVE(LINK_PRESENTATION_CHANGE_FOR_RADAR_115801517)
// FIXME: Remove this once HAVE(LINK_PRESENTATION_CHANGE_FOR_RADAR_115801517) is true
// for all platforms where HAVE(NSURL_TITLE) is set.
@interface NSURL ()
@property (nonatomic, copy, setter=_setTitle:) NSString *_title;
@end
#endif

@implementation NSURL (WebKitUtilities)

- (void)_web_setTitle:(NSString *)title
{
#if HAVE(LINK_PRESENTATION_CHANGE_FOR_RADAR_115801517)
    // -[LPLinkMetadata setTitle:] additionally sets the `_title` SPI attribute on
    // the NSURL URL in OS versions where this codepath is compiled.
    auto metadata = adoptNS([PAL::allocLPLinkMetadataInstance() init]);
    [metadata setURL:self];
    [metadata setTitle:title];
#else
    self._title = title;
#endif
}

- (NSString *)_web_title
{
#if HAVE(LINK_PRESENTATION_CHANGE_FOR_RADAR_115801517)
    // -[LPLinkMetadata title] falls back to the `_title` SPI attribute on the NSURL
    // in OS versions where this codepath is compiled.
    auto metadata = adoptNS([PAL::allocLPLinkMetadataInstance() init]);
    [metadata setURL:self];
    return [metadata title];
#else
    return self._title;
#endif
}

@end

#endif // HAVE(NSURL_TITLE)
