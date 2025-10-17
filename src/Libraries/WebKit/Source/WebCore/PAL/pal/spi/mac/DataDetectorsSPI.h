/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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
#import <wtf/Platform.h>

#if ENABLE(DATA_DETECTION)

#import <pal/spi/cocoa/DataDetectorsCoreSPI.h>
#import <wtf/SoftLinking.h>

#if PLATFORM(MAC)

#if USE(APPLE_INTERNAL_SDK)

// Can't include DDAction.h because as of this writing it is a private header that includes a non-private header with an "" include.
#import <DataDetectors/DDActionContext.h>
#import <DataDetectors/DDActionsManager.h>
#import <DataDetectors/DDHighlightDrawing.h>

#if HAVE(DATA_DETECTORS_MAC_ACTION)
#import <DataDetectors/DDMacAction.h>
#endif

#else // !USE(APPLE_INTERNAL_SDK)

#if HAVE(DATA_DETECTORS_MAC_ACTION)
@interface DDAction : NSObject
@property (readonly) NSString *actionUTI;
@end
@interface DDMacAction : DDAction
@end
#endif

@interface DDActionContext : NSObject <NSCopying, NSSecureCoding>

@property NSRect highlightFrame;
@property (retain) NSArray *allResults;
@property (retain) __attribute__((NSObject)) DDResultRef mainResult;
@property (assign) BOOL altMode;
@property (assign) BOOL immediate;

@property (retain) NSPersonNameComponents *authorNameComponents;

@property (copy) NSArray *allowedActionUTIs;

- (DDActionContext *)contextForView:(NSView *)view altMode:(BOOL)altMode interactionStartedHandler:(void (^)(void))interactionStartedHandler interactionChangedHandler:(void (^)(void))interactionChangedHandler interactionStoppedHandler:(void (^)(void))interactionStoppedHandler;

@end

#if HAVE(SECURE_ACTION_CONTEXT)
@interface DDSecureActionContext : DDActionContext
@end
#endif

@interface DDActionsManager : NSObject

+ (DDActionsManager *)sharedManager;
- (NSArray *)menuItemsForResult:(DDResultRef)result actionContext:(DDActionContext *)context;
- (NSArray *)menuItemsForTargetURL:(NSString *)targetURL actionContext:(DDActionContext *)context;
- (void)requestBubbleClosureUnanchorOnFailure:(BOOL)unanchorOnFailure;

+ (BOOL)shouldUseActionsWithContext:(DDActionContext *)context;
+ (void)didUseActions;

- (BOOL)hasActionsForResult:(DDResultRef)result actionContext:(DDActionContext *)actionContext;

- (NSArray *)menuItemsForValue:(NSString *)value type:(CFStringRef)type service:(NSString *)service context:(DDActionContext *)context;

@end

enum {
    DDHighlightStyleBubbleNone = 0,
    DDHighlightStyleBubbleStandard = 1
};

enum {
    DDHighlightStyleIconNone = (0 << 16),
    DDHighlightStyleStandardIconArrow = (1 << 16)
};

enum {
    DDHighlightStyleButtonShowAlways = (1 << 24),
};

#endif // !USE(APPLE_INTERNAL_SDK)

WTF_EXTERN_C_BEGIN
CFTypeID DDResultGetCFTypeID(void);
WTF_EXTERN_C_END

typedef struct __DDHighlight *DDHighlightRef;
typedef NSUInteger DDHighlightStyle;

#if !HAVE(DATA_DETECTORS_MAC_ACTION)

@interface DDAction : NSObject
@property (readonly) NSString *actionUTI;
@end

#endif // !HAVE(DATA_DETECTORS_MAC_ACTION)

#if HAVE(SECURE_ACTION_CONTEXT)
using WKDDActionContext = DDSecureActionContext;
#else
using WKDDActionContext = DDActionContext;
#endif

#endif // PLATFORM(MAC)

#endif // ENABLE(DATA_DETECTION)

