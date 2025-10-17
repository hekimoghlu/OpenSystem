/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#import <wtf/SoftLinking.h>

#if PLATFORM(IOS_FAMILY)

#import <UIKit/NSAttributedString.h>

SOFT_LINK_PRIVATE_FRAMEWORK(UIFoundation)

SOFT_LINK_CONSTANT(UIFoundation, NSFontAttributeName, NSString *)
#define NSFontAttributeName getNSFontAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSForegroundColorAttributeName, NSString *)
#define NSForegroundColorAttributeName getNSForegroundColorAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSBackgroundColorAttributeName, NSString *)
#define NSBackgroundColorAttributeName getNSBackgroundColorAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSStrokeColorAttributeName, NSString *)
#define NSStrokeColorAttributeName getNSStrokeColorAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSStrokeWidthAttributeName, NSString *)
#define NSStrokeWidthAttributeName getNSStrokeWidthAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSShadowAttributeName, NSString *)
#define NSShadowAttributeName getNSShadowAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSKernAttributeName, NSString *)
#define NSKernAttributeName getNSKernAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSLigatureAttributeName, NSString *)
#define NSLigatureAttributeName getNSLigatureAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSUnderlineStyleAttributeName, NSString *)
#define NSUnderlineStyleAttributeName getNSUnderlineStyleAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSStrikethroughStyleAttributeName, NSString *)
#define NSStrikethroughStyleAttributeName getNSStrikethroughStyleAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSBaselineOffsetAttributeName, NSString *)
#define NSBaselineOffsetAttributeName getNSBaselineOffsetAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSWritingDirectionAttributeName, NSString *)
#define NSWritingDirectionAttributeName getNSWritingDirectionAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSParagraphStyleAttributeName, NSString *)
#define NSParagraphStyleAttributeName getNSParagraphStyleAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSPresentationIntentAttributeName, NSString *)
#define NSPresentationIntentAttributeName getNSPresentationIntentAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSAttachmentAttributeName, NSString *)
#define NSAttachmentAttributeName getNSAttachmentAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSLinkAttributeName, NSString *)
#define NSLinkAttributeName getNSLinkAttributeName()
SOFT_LINK_CONSTANT(UIFoundation, NSAuthorDocumentAttribute, NSString *)
#define NSAuthorDocumentAttribute getNSAuthorDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSEditorDocumentAttribute, NSString *)
#define NSEditorDocumentAttribute getNSEditorDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSGeneratorDocumentAttribute, NSString *)
#define NSGeneratorDocumentAttribute getNSGeneratorDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSCompanyDocumentAttribute, NSString *)
#define NSCompanyDocumentAttribute getNSCompanyDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSDisplayNameDocumentAttribute, NSString *)
#define NSDisplayNameDocumentAttribute getNSDisplayNameDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSCopyrightDocumentAttribute, NSString *)
#define NSCopyrightDocumentAttribute getNSCopyrightDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSSubjectDocumentAttribute, NSString *)
#define NSSubjectDocumentAttribute getNSSubjectDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSCommentDocumentAttribute, NSString *)
#define NSCommentDocumentAttribute getNSCommentDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSNoIndexDocumentAttribute, NSString *)
#define NSNoIndexDocumentAttribute getNSNoIndexDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSKeywordsDocumentAttribute, NSString *)
#define NSKeywordsDocumentAttribute getNSKeywordsDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSCreationTimeDocumentAttribute, NSString *)
#define NSCreationTimeDocumentAttribute getNSCreationTimeDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSModificationTimeDocumentAttribute, NSString *)
#define NSModificationTimeDocumentAttribute getNSModificationTimeDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSConvertedDocumentAttribute, NSString *)
#define NSConvertedDocumentAttribute getNSConvertedDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSCocoaVersionDocumentAttribute, NSString *)
#define NSCocoaVersionDocumentAttribute getNSCocoaVersionDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSBackgroundColorDocumentAttribute, NSString *)
#define NSBackgroundColorDocumentAttribute getNSBackgroundColorDocumentAttribute()
SOFT_LINK_CONSTANT(UIFoundation, NSMarkedClauseSegmentAttributeName, NSString *)
#define NSMarkedClauseSegmentAttributeName getNSMarkedClauseSegmentAttributeName()

#import <UIKit/NSTextList.h>
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerCircle, NSTextListMarkerFormat)
#define NSTextListMarkerCircle getNSTextListMarkerCircle()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerDisc, NSTextListMarkerFormat)
#define NSTextListMarkerDisc getNSTextListMarkerDisc()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerSquare, NSTextListMarkerFormat)
#define NSTextListMarkerSquare getNSTextListMarkerSquare()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerLowercaseHexadecimal, NSTextListMarkerFormat)
#define NSTextListMarkerLowercaseHexadecimal getNSTextListMarkerLowercaseHexadecimal()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerUppercaseHexadecimal, NSTextListMarkerFormat)
#define NSTextListMarkerUppercaseHexadecimal getNSTextListMarkerUppercaseHexadecimal()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerOctal, NSTextListMarkerFormat)
#define NSTextListMarkerOctal getNSTextListMarkerOctal()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerLowercaseAlpha, NSTextListMarkerFormat)
#define NSTextListMarkerLowercaseAlpha getNSTextListMarkerLowercaseAlpha()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerUppercaseAlpha, NSTextListMarkerFormat)
#define NSTextListMarkerUppercaseAlpha getNSTextListMarkerUppercaseAlpha()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerLowercaseLatin, NSTextListMarkerFormat)
#define NSTextListMarkerLowercaseLatin getNSTextListMarkerLowercaseLatin()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerUppercaseLatin, NSTextListMarkerFormat)
#define NSTextListMarkerUppercaseLatin getNSTextListMarkerUppercaseLatin()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerLowercaseRoman, NSTextListMarkerFormat)
#define NSTextListMarkerLowercaseRoman getNSTextListMarkerLowercaseRoman()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerUppercaseRoman, NSTextListMarkerFormat)
#define NSTextListMarkerUppercaseRoman getNSTextListMarkerUppercaseRoman()
SOFT_LINK_CONSTANT(UIFoundation, NSTextListMarkerDecimal, NSTextListMarkerFormat)
#define NSTextListMarkerDecimal getNSTextListMarkerDecimal()

// We don't softlink NSSuperscriptAttributeName because UIFoundation stopped exporting it.
// This attribute is being deprecated at the API level, but internally UIFoundation
// will continue to support it.
static NSString *const NSSuperscriptAttributeName = @"NSSuperscript";

static NSString *const NSExcludedElementsDocumentAttribute = @"ExcludedElements";

@interface NSAttributedString ()
- (id)initWithRTF:(NSData *)data documentAttributes:(NSDictionary **)dict;
- (id)initWithRTFD:(NSData *)data documentAttributes:(NSDictionary **)dict;
- (NSData *)RTFFromRange:(NSRange)range documentAttributes:(NSDictionary *)dict;
- (NSData *)RTFDFromRange:(NSRange)range documentAttributes:(NSDictionary *)dict;
- (BOOL)containsAttachments;
@end

#endif // PLATFORM(IOS_FAMILY)
