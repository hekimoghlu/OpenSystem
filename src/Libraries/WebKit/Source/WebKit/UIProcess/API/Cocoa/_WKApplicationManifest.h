/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#import <Foundation/Foundation.h>
#import <WebKit/WKFoundation.h>

#if TARGET_OS_IPHONE
@class UIColor;
#else
@class NSColor;
#endif
@class _WKApplicationManifestIcon;
@class _WKApplicationManifestShortcut;

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, _WKApplicationManifestDirection) {
    _WKApplicationManifestDirectionAuto,
    _WKApplicationManifestDirectionLTR,
    _WKApplicationManifestDirectionRTL,
} WK_API_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2));

typedef NS_ENUM(NSInteger, _WKApplicationManifestDisplayMode) {
    _WKApplicationManifestDisplayModeBrowser,
    _WKApplicationManifestDisplayModeMinimalUI,
    _WKApplicationManifestDisplayModeStandalone,
    _WKApplicationManifestDisplayModeFullScreen,
} WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

typedef NS_ENUM(NSInteger, _WKApplicationManifestOrientation) {
    _WKApplicationManifestOrientationAny,
    _WKApplicationManifestOrientationLandscape,
    _WKApplicationManifestOrientationLandscapePrimary,
    _WKApplicationManifestOrientationLandscapeSecondary,
    _WKApplicationManifestOrientationNatural,
    _WKApplicationManifestOrientationPortrait,
    _WKApplicationManifestOrientationPortraitPrimary,
    _WKApplicationManifestOrientationPortraitSecondary,
} WK_API_AVAILABLE(macos(14.0), ios(17.0));

typedef NS_ENUM(NSInteger, _WKApplicationManifestIconPurpose) {
    _WKApplicationManifestIconPurposeAny = (1 << 0),
    _WKApplicationManifestIconPurposeMonochrome = (1 << 1),
    _WKApplicationManifestIconPurposeMaskable = (1 << 2),
} WK_API_AVAILABLE(macos(13.0), ios(16.0));

WK_CLASS_AVAILABLE(macos(10.13.4), ios(11.3))
@interface _WKApplicationManifest : NSObject <NSSecureCoding>

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithJSONData:(NSData *)jsonData manifestURL:(NSURL *)manifestURL documentURL:(NSURL *)documentURL WK_API_AVAILABLE(macos(14.5), ios(17.5), visionos(1.2));

@property (nonatomic, readonly, nullable, copy) NSString *rawJSON;
@property (nonatomic, readonly) _WKApplicationManifestDirection dir;
@property (nonatomic, readonly, nullable, copy) NSString *name;
@property (nonatomic, readonly, nullable, copy) NSString *shortName;
@property (nonatomic, readonly, nullable, copy) NSString *applicationDescription;
@property (nonatomic, readonly, nullable, copy) NSURL *scope;
@property (nonatomic, readonly) BOOL isDefaultScope;
@property (nonatomic, readonly, copy) NSURL *manifestURL;
@property (nonatomic, readonly, copy) NSURL *startURL;
@property (nonatomic, readonly, copy) NSURL *manifestId WK_API_AVAILABLE(macos(13.3), ios(16.4));
@property (nonatomic, readonly) _WKApplicationManifestDisplayMode displayMode;
@property (nonatomic, readonly, copy) NSArray<NSString *> *categories WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));
@property (nonatomic, readonly, copy) NSArray<_WKApplicationManifestIcon *> *icons WK_API_AVAILABLE(macos(13.0), ios(16.0));
@property (nonatomic, readonly, copy) NSArray<_WKApplicationManifestShortcut *> *shortcuts WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));

#if TARGET_OS_IPHONE
@property (nonatomic, readonly, nullable, copy) UIColor *backgroundColor WK_API_AVAILABLE(ios(17.0));
#else
@property (nonatomic, readonly, nullable, copy) NSColor *backgroundColor WK_API_AVAILABLE(macos(14.0));
#endif

#if TARGET_OS_IPHONE
@property (nonatomic, readonly, nullable, copy) UIColor *themeColor WK_API_AVAILABLE(ios(15.0));
#else
@property (nonatomic, readonly, nullable, copy) NSColor *themeColor WK_API_AVAILABLE(macos(12.0));
#endif

+ (_WKApplicationManifest *)applicationManifestFromJSON:(NSString *)json manifestURL:(nullable NSURL *)manifestURL documentURL:(nullable NSURL *)documentURL;

@end

WK_CLASS_AVAILABLE(macos(13.0), ios(16.0))
@interface _WKApplicationManifestIcon : NSObject <NSSecureCoding>

@property (nonatomic, readonly, copy) NSURL *src;
@property (nonatomic, readonly, copy) NSArray<NSString *> *sizes;
@property (nonatomic, readonly, copy) NSString *type;
@property (nonatomic, readonly) NSArray<NSNumber *> *purposes;

@end

WK_CLASS_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1))
@interface _WKApplicationManifestShortcut : NSObject <NSSecureCoding>

@property (nonatomic, readonly, copy) NSString *name;
@property (nonatomic, readonly, copy) NSURL *url;
@property (nonatomic, readonly, copy) NSArray<_WKApplicationManifestIcon *> *icons;

@end

NS_ASSUME_NONNULL_END
