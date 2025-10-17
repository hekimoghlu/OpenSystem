/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

/**
 * WKInputType exposes a subset of all known input types enumerated in
 * WebKit::InputType. While there is currently a one-to-one mapping, we
 * need to consider how we should expose certain input types.
 */
typedef NS_ENUM(NSInteger, WKInputType) {
    WKInputTypeNone,
    WKInputTypeContentEditable,
    WKInputTypeText,
    WKInputTypePassword,
    WKInputTypeTextArea,
    WKInputTypeSearch,
    WKInputTypeEmail,
    WKInputTypeURL,
    WKInputTypePhone,
    WKInputTypeNumber,
    WKInputTypeNumberPad,
    WKInputTypeDate,
    WKInputTypeDateTime,
    WKInputTypeDateTimeLocal,
    WKInputTypeMonth,
    WKInputTypeWeek,
    WKInputTypeTime,
    WKInputTypeSelect,
    WKInputTypeColor,
    WKInputTypeDrawing,
};

/**
 * The _WKFocusedElementInfo provides basic information about an element that
 * has been focused (either programmatically or through user interaction) but
 * is not causing any input UI (e.g. keyboard, date picker, etc.) to be shown.
 */
@protocol _WKFocusedElementInfo <NSObject>

/* The type of the input element that was focused. */
@property (nonatomic, readonly) WKInputType type;

/* The value of the input at the time it was focused. */
@property (nonatomic, readonly, copy) NSString *value;

/* The placeholder text of the input. */
@property (nonatomic, readonly, copy) NSString *placeholder;

/* The text of a label element associated with the input. */
@property (nonatomic, readonly, copy) NSString *label;

/**
 * Whether the element was focused due to user interaction. NO indicates that
 * the element was focused programmatically, e.g. by calling focus() in JavaScript
 * or by using the autofocus attribute.
 */
@property (nonatomic, readonly, getter=isUserInitiated) BOOL userInitiated;

@property (nonatomic, readonly) NSObject <NSSecureCoding> *userObject WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

@end
