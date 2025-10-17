/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#if ENABLE(DATA_DETECTION)

typedef struct __DDResult *DDResultRef;

#if USE(APPLE_INTERNAL_SDK)

#import <DataDetectorsCore/DDBinderKeys_Private.h>
#import <DataDetectorsCore/DDScanQuery_Private.h>
#import <DataDetectorsCore/DDScanner.h>
#import <DataDetectorsCore/DDScannerResult.h>
#import <DataDetectorsCore/DataDetectorsCore.h>

#if PLATFORM(IOS_FAMILY)
#import <DataDetectorsCore/DDOptionalSource.h>
#import <DataDetectorsCore/DDURLifier.h>
#endif // PLATFORM(IOS_FAMILY)

#else // !USE(APPLE_INTERNAL_SDK)

#import <wtf/Compiler.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#import <Foundation/Foundation.h>

typedef enum {
    DDScannerTypeStandard = 0,
    DDScannerType1 = 1,
    DDScannerType2 = 2,
} DDScannerType;

enum {
    DDScannerCopyResultsOptionsNone = 0,
    DDScannerCopyResultsOptionsNoOverlap = 1 << 0,
    DDScannerCopyResultsOptionsCoalesceSignatures = 1 << 1,
};

typedef CFIndex DDScannerSource;

enum {
    DDURLifierPhoneNumberDetectionNone = 0,
    DDURLifierPhoneNumberDetectionRegular = 1 << 1,
    DDURLifierPhoneNumberDetectionQuotedShorts = 1 << 2,
    DDURLifierPhoneNumberDetectionUnquotedShorts = 1 << 3
};
typedef NSUInteger DDURLifierPhoneNumberDetectionTypes;

typedef enum __DDTextCoalescingType {
    DDTextCoalescingTypeNone = 0,
    DDTextCoalescingTypeSpace = 1,
    DDTextCoalescingTypeTab = 2,
    DDTextCoalescingTypeLineBreak = 3,
    DDTextCoalescingTypeHardBreak = 4,
} DDTextCoalescingType;

typedef enum {
    DDResultCategoryUnknown = 0,
    DDResultCategoryLink = 1,
    DDResultCategoryPhoneNumber = 2,
    DDResultCategoryAddress = 3,
    DDResultCategoryCalendarEvent = 4,
    DDResultCategoryMisc = 5,
} DDResultCategory;

typedef enum __DDTextFragmentType {
    DDTextFragmentTypeTrimWhiteSpace =  0x1,
    DDTextFragmentTypeIgnoreCRLF =  0x2,
} DDTextFragmentMode;

#if HAVE(DDSCANNER_QOS_CONFIGURATION)
typedef enum __DDQOS {
    DDQOSRegular = 0
    DDQOSEnhanced = 2,
    DDQOSHighest = 4,
} DDQOS;
#endif

extern CFStringRef const DDBinderHttpURLKey;
extern CFStringRef const DDBinderWebURLKey;
extern CFStringRef const DDBinderMailURLKey;
extern CFStringRef const DDBinderGenericURLKey;
extern CFStringRef const DDBinderEmailKey;
extern CFStringRef const DDBinderTrackingNumberKey;
extern CFStringRef const DDBinderFlightInformationKey;
extern CFStringRef const DDBinderParsecSourceKey;
extern CFStringRef const DDBinderSignatureBlockKey;

@interface DDScannerResult : NSObject <NSCoding, NSSecureCoding>
@property (readonly, nonatomic) NSRange urlificationRange;
+ (NSArray *)resultsFromCoreResults:(CFArrayRef)coreResults;
- (DDResultRef)coreResult;
@end

#define DDResultPropertyPassiveDisplay   (1 << 0)

typedef struct __DDQueryOffset {
    CFIndex queryIndex:32;
    CFIndex offset:32;
} DDQueryOffset;

typedef struct __DDQueryRange {
    DDQueryOffset start;
    DDQueryOffset end;
} DDQueryRange;

typedef struct __DDQueryFragment {
    CFStringRef string;
    void *identifier;
    CFRange range;
    CFIndex absoluteOffset;
    CFIndex contextOffset:26;
    DDTextCoalescingType coalescing:3;
    DDTextFragmentMode mode:2;
    Boolean lineBreakDoesNotCoalesce:1;
} DDQueryFragment;

struct __DDScanQuery {
    uint8_t _cfBase[16]; // 16 bytes; the size of the real type, CFRuntimeBase.
    DDQueryFragment *fragments;
    CFIndex capacity;
    CFIndex numberOfFragments;
    void (*releaseCallBack)(void * context, void * identifier);
    void *context;
};

#endif // !USE(APPLE_INTERNAL_SDK)

static_assert(sizeof(DDQueryOffset) == 8, "DDQueryOffset is no longer 8 bytes. Update the definition of DDQueryOffset in this file to match the new size.");

typedef struct __DDScanQuery *DDScanQueryRef;
typedef struct __DDScanner *DDScannerRef;

#if !USE(APPLE_INTERNAL_SDK)
static inline DDQueryFragment *DDScanQueryGetFragmentAtIndex(DDScanQueryRef query, CFIndex anIndex)
{
    return &query->fragments[anIndex];
}

static inline CFIndex _DDScanQueryGetNumberOfFragments(DDScanQueryRef query)
{
    return query->numberOfFragments;
}

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#endif

typedef CFIndex DDScannerCopyResultsOptions;
typedef CFIndex DDScannerOptions;

enum {
    DDScannerSourceSpotlight = 1<<1,
};

WTF_EXTERN_C_BEGIN

extern const DDScannerCopyResultsOptions DDScannerCopyResultsOptionsForPassiveUse;

DDScannerRef DDScannerCreate(DDScannerType, DDScannerOptions, CFErrorRef*);
DDScanQueryRef DDScanQueryCreate(CFAllocatorRef);
DDScanQueryRef DDScanQueryCreateFromString(CFAllocatorRef, CFStringRef, CFRange);
Boolean DDScannerScanQuery(DDScannerRef, DDScanQueryRef);
CFArrayRef DDScannerCopyResultsWithOptions(DDScannerRef, DDScannerCopyResultsOptions);
CFRange DDResultGetRange(DDResultRef);
CFStringRef DDResultGetType(DDResultRef);
DDResultCategory DDResultGetCategory(DDResultRef);
Boolean DDResultIsPastDate(DDResultRef, CFDateRef referenceDate, CFTimeZoneRef referenceTimeZone);
void DDScanQueryAddTextFragment(DDScanQueryRef, CFStringRef, CFRange, void *identifier, DDTextFragmentMode, DDTextCoalescingType);
void DDScanQueryAddSeparator(DDScanQueryRef, DDTextCoalescingType);
void DDScanQueryAddLineBreak(DDScanQueryRef);
void *DDScanQueryGetFragmentMetaData(DDScanQueryRef, CFIndex queryIndex);
bool DDResultHasProperties(DDResultRef, CFIndex propertySet);
CFArrayRef DDResultGetSubResults(DDResultRef);
DDQueryRange DDResultGetQueryRangeForURLification(DDResultRef);
void DDResultDisableURLSchemeChecking();

#if HAVE(DDSCANNER_QOS_CONFIGURATION)
void DDScannerSetQOS(DDScannerRef, DDQOS);
#endif

WTF_EXTERN_C_END

#endif
