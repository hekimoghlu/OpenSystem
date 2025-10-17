/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

#include <wtf/text/WTFString.h>

namespace WebDriver {

enum class MouseButton { None, Left, Middle, Right };
enum class PointerType { Mouse, Pen, Touch };

struct InputSource {
    enum class Type { None, Key, Pointer, Wheel };

    Type type;
    std::optional<PointerType> pointerType;
};

struct PointerParameters {
    PointerType pointerType { PointerType::Mouse };
};

struct PointerOrigin {
    enum class Type { Viewport, Pointer, Element };

    Type type;
    std::optional<String> elementID;
};

struct Action {
    enum class Type { None, Key, Pointer, Wheel };
    enum class Subtype { Pause, PointerUp, PointerDown, PointerMove, PointerCancel, KeyUp, KeyDown, Scroll };

    Action(const String& id, Type type, Subtype subtype)
        : id(id)
        , type(type)
        , subtype(subtype)
    {
    }

    String id;
    Type type;
    Subtype subtype;
    std::optional<unsigned> duration;

    std::optional<PointerType> pointerType;
    std::optional<MouseButton> button;
    std::optional<PointerOrigin> origin;
    std::optional<int64_t> x;
    std::optional<int64_t> y;
    std::optional<int64_t> deltaX;
    std::optional<int64_t> deltaY;

    std::optional<String> key;
};

} // WebDriver
