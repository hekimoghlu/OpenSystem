/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
#import <Availability.h>
#import <TargetConditionals.h>

#ifdef __cplusplus
#define WK_EXTERN extern "C" __attribute__((visibility ("default")))
#else
#define WK_EXTERN extern __attribute__((visibility ("default")))
#endif

#ifdef NS_SWIFT_ASYNC_NAME
#define WK_SWIFT_ASYNC_NAME(...) NS_SWIFT_ASYNC_NAME(__VA_ARGS__)
#else
#define WK_SWIFT_ASYNC_NAME(...)
#endif

#ifdef NS_SWIFT_ASYNC
#define WK_SWIFT_ASYNC(...) NS_SWIFT_ASYNC(__VA_ARGS__)
#else
#define WK_SWIFT_ASYNC(...)
#endif

#ifdef NS_SWIFT_ASYNC_THROWS_ON_FALSE
#define WK_SWIFT_ASYNC_THROWS_ON_FALSE(...) NS_SWIFT_ASYNC_THROWS_ON_FALSE(__VA_ARGS__)
#else
#define WK_SWIFT_ASYNC_THROWS_ON_FALSE(...)
#endif

#if __has_attribute(swift_async_error)
#define WK_NULLABLE_RESULT _Nullable_result
#else
#define WK_NULLABLE_RESULT _Nullable
#endif

#ifdef NS_SWIFT_UI_ACTOR
#define WK_SWIFT_UI_ACTOR NS_SWIFT_UI_ACTOR
#else
#define WK_SWIFT_UI_ACTOR
#endif

#ifdef NS_HEADER_AUDIT_BEGIN
#define WK_HEADER_AUDIT_BEGIN NS_HEADER_AUDIT_BEGIN
#define WK_HEADER_AUDIT_END NS_HEADER_AUDIT_END
#else
#define WK_HEADER_AUDIT_BEGIN(...) NS_ASSUME_NONNULL_BEGIN
#define WK_HEADER_AUDIT_END(...) NS_ASSUME_NONNULL_END
#endif

#ifndef WK_FRAMEWORK_HEADER_POSTPROCESSING_ENABLED

#define WK_API_AVAILABLE(...)
#define WK_API_UNAVAILABLE(...)
#define WK_CLASS_AVAILABLE(...) __attribute__((visibility("default"))) WK_API_AVAILABLE(__VA_ARGS__)
#define WK_API_DEPRECATED(_message, ...) __attribute__((deprecated(_message)))
#define WK_API_DEPRECATED_WITH_REPLACEMENT(_replacement, ...) __attribute__((deprecated("use " #_replacement)))
#define WK_CLASS_DEPRECATED_WITH_REPLACEMENT(_replacement, ...) __attribute__((visibility("default"))) __attribute__((deprecated("use " #_replacement)))

#endif
