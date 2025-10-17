/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

#include "ContentsFormat.h"
#include <wtf/Forward.h>

#if PLATFORM(MAC)
OBJC_CLASS NSScreen;
OBJC_CLASS NSWindow;
#ifdef NSGEOMETRY_TYPES_SAME_AS_CGGEOMETRY_TYPES
typedef struct CGRect NSRect;
typedef struct CGPoint NSPoint;
#else
typedef struct _NSRect NSRect;
typedef struct _NSPoint NSPoint;
#endif
#endif

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS UIScreen;
#endif

#if USE(CG)
typedef struct CGColorSpace *CGColorSpaceRef;
#endif

// X11 headers define a bunch of macros with common terms, interfering with WebCore and WTF enum values.
// As a workaround, we explicitly undef them here.
#if defined(None)
#undef None
#endif

namespace WebCore {

class DestinationColorSpace;
class FloatPoint;
class FloatRect;
class FloatSize;
class PlatformCALayerClient;
class Widget;

using PlatformDisplayID = uint32_t;

using PlatformGPUID = uint64_t; // On MAC, global IOKit registryID that can identify a GPU across process boundaries.

int screenDepth(Widget*);
int screenDepthPerComponent(Widget*);
bool screenIsMonochrome(Widget*);
WEBCORE_EXPORT DestinationColorSpace screenColorSpace(Widget* = nullptr);

bool screenHasInvertedColors();

#if USE(GLIB)
double fontDPI(); // dpi to use for font scaling
double screenDPI(PlatformDisplayID); // dpi of the display device, corrected for device scaling
#endif

FloatRect screenRect(Widget*);
FloatRect screenAvailableRect(Widget*);

WEBCORE_EXPORT ContentsFormat screenContentsFormat(Widget* = nullptr, PlatformCALayerClient* = nullptr);
WEBCORE_EXPORT bool screenSupportsExtendedColor(Widget* = nullptr);

enum class DynamicRangeMode : uint8_t {
    None,
    Standard,
    HLG,
    HDR10,
    DolbyVisionPQ,
};
#if HAVE(AVPLAYER_VIDEORANGEOVERRIDE)
WEBCORE_EXPORT DynamicRangeMode preferredDynamicRangeMode(Widget* = nullptr);
#else
constexpr DynamicRangeMode preferredDynamicRangeMode(Widget* = nullptr) { return DynamicRangeMode::Standard; }
#endif

#if PLATFORM(MAC) || PLATFORM(IOS_FAMILY)
WEBCORE_EXPORT bool screenSupportsHighDynamicRange(Widget* = nullptr);
#else
constexpr bool screenSupportsHighDynamicRange(Widget* = nullptr) { return false; }
#endif

struct ScreenProperties;
struct ScreenData;
    
WEBCORE_EXPORT ScreenProperties collectScreenProperties();
WEBCORE_EXPORT void setScreenProperties(const ScreenProperties&);
const ScreenProperties& getScreenProperties();
WEBCORE_EXPORT const ScreenData* screenData(PlatformDisplayID screendisplayID);
WEBCORE_EXPORT PlatformDisplayID primaryScreenDisplayID();
    
#if PLATFORM(MAC)

WEBCORE_EXPORT PlatformDisplayID displayID(NSScreen *);

WEBCORE_EXPORT NSScreen *screen(NSWindow *);
NSScreen *screen(PlatformDisplayID);

WEBCORE_EXPORT FloatRect screenRectForDisplay(PlatformDisplayID);
WEBCORE_EXPORT FloatRect screenRectForPrimaryScreen();
WEBCORE_EXPORT FloatRect availableScreenRect(NSScreen *);

WEBCORE_EXPORT FloatRect toUserSpace(const NSRect&, NSWindow *destination);
WEBCORE_EXPORT FloatRect toUserSpaceForPrimaryScreen(const NSRect&);
WEBCORE_EXPORT FloatPoint toUserSpaceForPrimaryScreen(const NSPoint&);
WEBCORE_EXPORT NSRect toDeviceSpace(const FloatRect&, NSWindow *source);

NSPoint flipScreenPoint(const NSPoint&, NSScreen *);

WEBCORE_EXPORT void setShouldOverrideScreenSupportsHighDynamicRange(bool shouldOverride, bool supportsHighDynamicRange);

uint32_t primaryOpenGLDisplayMask();
uint32_t displayMaskForDisplay(PlatformDisplayID);

PlatformGPUID primaryGPUID();
PlatformGPUID gpuIDForDisplay(PlatformDisplayID);
PlatformGPUID gpuIDForDisplayMask(uint32_t);

WEBCORE_EXPORT FloatRect safeScreenFrame(NSScreen *);

#endif // !PLATFORM(MAC)

#if PLATFORM(IOS_FAMILY)

float screenPPIFactor();
WEBCORE_EXPORT FloatSize screenSize();
WEBCORE_EXPORT FloatSize availableScreenSize();
WEBCORE_EXPORT FloatSize overrideScreenSize();
WEBCORE_EXPORT FloatSize overrideAvailableScreenSize();
WEBCORE_EXPORT float screenScaleFactor(UIScreen * = nullptr);

#endif

#if ENABLE(TOUCH_EVENTS)
#if PLATFORM(GTK) || PLATFORM(WPE)
WEBCORE_EXPORT bool screenHasTouchDevice();
WEBCORE_EXPORT bool screenIsTouchPrimaryInputDevice();
#else
constexpr bool screenHasTouchDevice() { return true; }
constexpr bool screenIsTouchPrimaryInputDevice() { return true; }
#endif
#endif

} // namespace WebCore
