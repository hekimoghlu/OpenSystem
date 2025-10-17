/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
#ifndef _IOKIT_IOGRAPHICSDEVICE_H
#define _IOKIT_IOGRAPHICSDEVICE_H

#include <IOKit/IOService.h>

#include <IOKit/graphics/IOGraphicsTypes.h>


class IOGraphicsDevice : public IOService
{
    OSDeclareAbstractStructors(IOGraphicsDevice)

public:

    virtual void hideCursor( void ) = 0;
    virtual void showCursor( IOGPoint * cursorLoc, int frame ) = 0;
    virtual void moveCursor( IOGPoint * cursorLoc, int frame ) = 0;

    virtual void getVBLTime( AbsoluteTime * time, AbsoluteTime * delta ) = 0;

    virtual void getBoundingRect ( IOGBounds ** bounds ) = 0;
};

#endif /* ! _IOKIT_IOGRAPHICSDEVICE_H */

