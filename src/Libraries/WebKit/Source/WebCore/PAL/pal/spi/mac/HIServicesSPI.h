/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#pragma once

#include <pal/spi/cg/CoreGraphicsSPI.h>
#include <span>
#include <wtf/StdLibExtras.h>

#if USE(APPLE_INTERNAL_SDK)

#include <ApplicationServices/ApplicationServicesPriv.h>

#else

#include <ApplicationServices/ApplicationServices.h>

typedef CF_ENUM(SInt32, CoreCursorType) {
    kCoreCursorFirstCursor = 0,
    kCoreCursorArrow = kCoreCursorFirstCursor,
    kCoreCursorIBeam,
    kCoreCursorMakeAlias,
    kCoreCursorNotAllowed,
    kCoreCursorBusyButClickable,
    kCoreCursorCopy,
    kCoreCursorScreenShotSelection = 7,
    kCoreCursorScreenShotSelectionToClip,
    kCoreCursorScreenShotWindow,
    kCoreCursorScreenShotWindowToClip,
    kCoreCursorClosedHand,
    kCoreCursorOpenHand,
    kCoreCursorPointingHand,
    kCoreCursorCountingUpHand,
    kCoreCursorCountingDownHand,
    kCoreCursorCountingUpAndDownHand,
    kCoreCursorResizeLeft,
    kCoreCursorResizeRight,
    kCoreCursorResizeLeftRight,
    kCoreCursorCross,
    kCoreCursorResizeUp,
    kCoreCursorResizeDown,
    kCoreCursorResizeUpDown,
    kCoreCursorContextualMenu,
    kCoreCursorPoof,
    kCoreCursorIBeamVertical,
    kCoreCursorWindowResizeEast,
    kCoreCursorWindowResizeEastWest,
    kCoreCursorWindowResizeNorthEast,
    kCoreCursorWindowResizeNorthEastSouthWest,
    kCoreCursorWindowResizeNorth,
    kCoreCursorWindowResizeNorthSouth,
    kCoreCursorWindowResizeNorthWest,
    kCoreCursorWindowResizeNorthWestSouthEast,
    kCoreCursorWindowResizeSouthEast,
    kCoreCursorWindowResizeSouth,
    kCoreCursorWindowResizeSouthWest,
    kCoreCursorWindowResizeWest,
    kCoreCursorWindowMove,
    kCoreCursorHelp,
    kCoreCursorCell,
    kCoreCursorZoomIn,
    kCoreCursorZoomOut,
    kCoreCursorLastCursor = kCoreCursorZoomOut
};

enum {
    kCoreDragImageSpecVersionOne = 1,
};

struct CoreDragImageSpec {
    UInt32 version;
    SInt32 pixelsWide;
    SInt32 pixelsHigh;
    SInt32 bitsPerSample;
    SInt32 samplesPerPixel;
    SInt32 bitsPerPixel;
    SInt32 bytesPerRow;
    Boolean isPlanar;
    Boolean hasAlpha;
    const UInt8* data[5];
};

enum {
    kMSHDoNotCreateSendRightOption = 0x4,
};

typedef UInt32 MSHCreateOptions;
typedef const struct __AXTextMarker* AXTextMarkerRef;
typedef const struct __AXTextMarkerRange* AXTextMarkerRangeRef;
typedef struct CoreDragImageSpec CoreDragImageSpec;
typedef struct OpaqueCoreDrag* CoreDragRef;

WTF_EXTERN_C_BEGIN

AXTextMarkerRangeRef AXTextMarkerRangeCreate(CFAllocatorRef, AXTextMarkerRef startMarker, AXTextMarkerRef endMarker);
AXTextMarkerRef AXTextMarkerCreate(CFAllocatorRef, const UInt8* bytes, CFIndex length);
AXTextMarkerRef AXTextMarkerRangeCopyStartMarker(AXTextMarkerRangeRef);
AXTextMarkerRef AXTextMarkerRangeCopyEndMarker(AXTextMarkerRangeRef);
CFIndex AXTextMarkerGetLength(AXTextMarkerRef);
CFRunLoopSourceRef MSHCreateMIGServerSource(CFAllocatorRef, CFIndex order, mig_subsystem_t sub_system, MSHCreateOptions, mach_port_t, void* user_data);
CFTypeID AXTextMarkerGetTypeID();
CFTypeID AXTextMarkerRangeGetTypeID();
CoreDragRef CoreDragGetCurrentDrag();
OSStatus CoreDragSetImage(CoreDragRef, CGPoint imageOffset, CoreDragImageSpec*, CGSRegionObj imageShape, float overallAlpha);
const UInt8* AXTextMarkerGetBytePtr(AXTextMarkerRef);
bool _AXUIElementRequestServicedBySecondaryAXThread(void);
OSStatus SetApplicationIsDaemon(Boolean);

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
AXError _AXUIElementUseSecondaryAXThread(bool enabled);
#endif

typedef CF_ENUM(int32_t, AXClientType)
{
    kAXClientTypeNoActiveRequestFound = 0,
    kAXClientTypeUnknown,
    kAXClientTypeRaft,
    kAXClientTypeXCUITest,
    kAXClientTypeXCTest,
    kAXClientTypeScripter2,
    kAXClientTypeSystemEvents,
    kAXClientTypeVoiceOver,
    kAXClientTypeAssistiveControl,
    kAXClientTypeFullKeyboardAccess,
    kAXClientTypeDictation,
};
AXClientType _AXGetClientForCurrentRequestUntrusted(void);
void _AXSetClientIdentificationOverride(AXClientType);

extern CFStringRef kAXInterfaceReduceMotionKey;
extern CFStringRef kAXInterfaceReduceMotionStatusDidChangeNotification;

extern CFStringRef kAXInterfaceIncreaseContrastKey;

extern CFStringRef kAXInterfaceDifferentiateWithoutColorKey;

WTF_EXTERN_C_END

#endif // USE(APPLE_INTERNAL_SDK)

#if PLATFORM(MAC)
inline std::span<const uint8_t> AXTextMarkerGetByteSpan(AXTextMarkerRef marker)
{
    return unsafeMakeSpan(AXTextMarkerGetBytePtr(marker), AXTextMarkerGetLength(marker));
}
#endif

WTF_EXTERN_C_BEGIN

typedef Boolean (*AXAuditTokenIsAuthenticatedCallback)(audit_token_t);

WTF_EXTERN_C_END

#define kAXClientTypeWebKitTesting 999999
