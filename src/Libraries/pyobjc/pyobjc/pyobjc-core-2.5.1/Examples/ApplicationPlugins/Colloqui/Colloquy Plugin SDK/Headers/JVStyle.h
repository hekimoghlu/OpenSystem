/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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

#import <Foundation/NSObject.h>

@class NSBundle;
@class NSDictionary;
@class NSArray;
@class NSSet;
@class NSString;
@class NSURL;

extern NSString *JVStylesScannedNotification;
extern NSString *JVDefaultStyleChangedNotification;
extern NSString *JVDefaultStyleVariantChangedNotification;
extern NSString *JVNewStyleVariantAddedNotification;
extern NSString *JVStyleVariantChangedNotification;

@interface JVStyle : NSObject {
	NSBundle *_bundle;
	NSDictionary *_parameters;
	NSArray *_styleOptions;
	NSArray *_variants;
	NSArray *_userVariants;
	void *_XSLStyle; /* xsltStylesheet */
	BOOL _releasing;
}
+ (void) scanForStyles;
+ (NSSet *) styles;
+ (id) styleWithIdentifier:(NSString *) identifier;
+ (id) newWithBundle:(NSBundle *) bundle;

+ (id) defaultStyle;
+ (void) setDefaultStyle:(JVStyle *) style;

- (id) initWithBundle:(NSBundle *) bundle;

- (void) unlink;
- (void) reload;
- (BOOL) isCompliant;

- (NSBundle *) bundle;
- (NSString *) identifier;

- (NSString *) transformXML:(NSString *) xml withParameters:(NSDictionary *) parameters;
- (NSString *) transformXMLDocument:(/* xmlDoc */ void *) document withParameters:(NSDictionary *) parameters;

- (NSComparisonResult) compare:(JVStyle *) style;
- (NSString *) displayName;

- (NSString *) mainVariantDisplayName;
- (NSArray *) variantStyleSheetNames;
- (NSArray *) userVariantStyleSheetNames;
- (BOOL) isUserVariantName:(NSString *) name;
- (NSString *) defaultVariantName;
- (void) setDefaultVariantName:(NSString *) name;

- (NSArray *) styleSheetOptions;

- (void) setMainParameters:(NSDictionary *) parameters;
- (NSDictionary *) mainParameters;

- (NSURL *) baseLocation;
- (NSURL *) mainStyleSheetLocation;
- (NSURL *) variantStyleSheetLocationWithName:(NSString *) name;
- (NSString *) XMLStyleSheetFilePath;
- (NSString *) previewTranscriptFilePath;
- (NSString *) headerFilePath;

- (NSString *) contentsOfMainStyleSheet;
- (NSString *) contentsOfVariantStyleSheetWithName:(NSString *) name;
- (NSString *) contentsOfHeaderFile;
@end