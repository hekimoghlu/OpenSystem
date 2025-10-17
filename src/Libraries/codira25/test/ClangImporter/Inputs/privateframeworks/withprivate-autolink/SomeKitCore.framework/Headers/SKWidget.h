/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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

@import ObjectiveC;
@import Foundation;

@interface SKWidget : NSObject
- (void)someObjCMethod;
@end

@interface SKWidget(ObjCAPI)
- (void)someObjCExtensionMethod;
@property (readwrite,strong,nonnull) NSObject *anObject;
@end

@interface NSObject (SKWidget)
- (void)doSomethingWithWidget:(nonnull SKWidget *)widget;
@end

extern NSString * _Nonnull const SKWidgetErrorDomain;
typedef enum __attribute__((ns_error_domain(SKWidgetErrorDomain))) __attribute__((language_name("SKWidget.Error"))) SKWidgetErrorCode : NSInteger {
  SKWidgetErrorNone = 0,
  SKWidgetErrorBoom = 1
} SKWidgetErrorCode;

@interface SKWidget(Erroneous)
- (SKWidgetErrorCode)getCurrentError;
@end

extern void someKitGlobalFunc(void);
