/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
#ifndef IOHIDEventServiceTypes_h
#define IOHIDEventServiceTypes_h

/*!
 * @typedef IOHIDKeyboardEventOptions
 *
 * @abstract
 * Keyboard event options passed in to dispatchKeyboardEvent function in
 * IOHIDEventService.
 *
 * @field kIOHIDKeyboardEventOptionsNoKeyRepeat
 * Default behavior for keyboard events is to repeat keys if the key has been
 * held down for a certain amount of time defined in system preferences. Pass
 * in this option to not apply key repeat logic to this event.
 */
typedef enum {
    kIOHIDKeyboardEventOptionsNoKeyRepeat   = (1 << 8),
} IOHIDKeyboardEventOptions;

/*!
 * @typedef IOHIDPointerEventOptions
 *
 * @abstract
 * Pointer event options passed in to dispatch(Relative/Absolute)PointerEvent
 * function in IOHIDEventService.
 *
 * @field kIOHIDPointerEventOptionsNoAcceleration
 * Pointer events are subject to an acceleration algorithm. Pass in this option
 * if you do not wish to have acceleration logic applied to the pointer event.
 */
typedef enum {
    kIOHIDPointerEventOptionsNoAcceleration = (1 << 8),
} IOHIDPointerEventOptions;

/*!
 * @typedef IOHIDScrollEventOptions
 *
 * @abstract
 * Scroll event options passed in to dispatchScrollEvent function in
 * IOHIDEventService.
 *
 * @field kIOHIDScrollEventOptionsNoAcceleration
 * Scroll events are subject to an acceleration algorithm. Pass in this option
 * if you do not wish to have acceleration logic applied to the scroll event.
 */
typedef enum {
    kIOHIDScrollEventOptionsNoAcceleration  = (1 << 8),
} IOHIDScrollEventOptions;

#endif /* IOHIDEventServiceTypes_h */
