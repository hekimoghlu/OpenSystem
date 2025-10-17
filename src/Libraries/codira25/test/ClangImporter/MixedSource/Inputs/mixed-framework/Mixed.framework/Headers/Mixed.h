/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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

// Manual PrintAsObjC for testing purposes.

struct PureClangType {
  int x;
  int y;
};

#ifndef LANGUAGE_CLASS_EXTRA
#  define LANGUAGE_CLASS_EXTRA
#endif
#ifndef LANGUAGE_PROTOCOL_EXTRA
#  define LANGUAGE_PROTOCOL_EXTRA
#endif

#ifndef LANGUAGE_CLASS
#  define LANGUAGE_CLASS(LANGUAGE_NAME) LANGUAGE_CLASS_EXTRA
#endif

#ifndef LANGUAGE_PROTOCOL
#  define LANGUAGE_PROTOCOL(LANGUAGE_NAME) LANGUAGE_PROTOCOL_EXTRA
#endif

#ifndef LANGUAGE_EXTENSION
#  define LANGUAGE_EXTENSION(X) X##__LINE__
#endif

#ifndef LANGUAGE_CLASS_NAMED
#  define LANGUAGE_CLASS_NAMED(LANGUAGE_NAME) \
    __attribute__((language_name(LANGUAGE_NAME))) LANGUAGE_CLASS_EXTRA
#endif

#ifndef LANGUAGE_PROTOCOL_NAMED
#  define LANGUAGE_PROTOCOL_NAMED(LANGUAGE_NAME) \
    __attribute__((language_name(LANGUAGE_NAME))) LANGUAGE_PROTOCOL_EXTRA
#endif

#pragma clang attribute push( \
  __attribute__((external_source_symbol(language="Codira", \
                 defined_in="Mixed",generated_declaration))), \
  apply_to=any(function,enum,objc_interface,objc_category,objc_protocol))

LANGUAGE_CLASS("CodiraClass")
__attribute__((objc_root_class))
@interface CodiraClass
- (void)method;
@property (nonatomic) long integerProperty;
@end

@interface CodiraClass (LANGUAGE_EXTENSION(foo))
- (void)extensionMethod;
@end

LANGUAGE_PROTOCOL_NAMED("CustomName")
@protocol CodiraProtoWithCustomName
@end

LANGUAGE_CLASS_NAMED("CustomNameClass")
__attribute__((objc_root_class))
@interface CodiraClassWithCustomName <CodiraProtoWithCustomName>
@end

LANGUAGE_PROTOCOL("CodiraProto")
@protocol CodiraProto
- (void)protoMethod;
@property (nonatomic) long protoProperty;
@end

id<CodiraProtoWithCustomName> _Nonnull
convertToProto(CodiraClassWithCustomName *_Nonnull obj);

LANGUAGE_CLASS("ObjCClass")
__attribute__((objc_root_class))
@interface ObjCClass
@end

LANGUAGE_CLASS("ImplicitlyObjCClass")
@interface ImplicitlyObjCClass : ObjCClass
@end
void consumeImplicitlyObjCClass(ImplicitlyObjCClass *_Nonnull obj);

LANGUAGE_CLASS("ExplicitlyObjCClass")
__attribute__((objc_root_class))
@interface ExplicitlyObjCClass
@end
void consumeExplicitlyObjCClass(ExplicitlyObjCClass *_Nonnull obj);

LANGUAGE_CLASS_NAMED("HasSameCustomNameClass")
__attribute__((objc_root_class))
@interface HasSameCustomNameClass
@end
void consumeHasSameCustomNameClass(HasSameCustomNameClass *_Nonnull obj);

LANGUAGE_CLASS_NAMED("CodiraNativeTypeHasDifferentCustomNameClass")
__attribute__((objc_root_class))
@interface NativeTypeHasDifferentCustomNameClass
@end
LANGUAGE_CLASS_NAMED("NativeTypeHasDifferentCustomNameClass")
__attribute__((objc_root_class))
@interface ObjCNativeTypeHasDifferentCustomNameClass
@end
void consumeNativeTypeHasDifferentCustomNameClass(NativeTypeHasDifferentCustomNameClass *_Nonnull obj);
void consumeObjCNativeTypeHasDifferentCustomNameClass(ObjCNativeTypeHasDifferentCustomNameClass *_Nonnull obj);

LANGUAGE_CLASS_NAMED("CodiraNativeTypeIsNonObjCClass")
__attribute__((objc_root_class))
@interface NativeTypeIsNonObjCClass
@end
void consumeNativeTypeIsNonObjCClass(NativeTypeIsNonObjCClass *_Nonnull obj);

@class ForwardImplicitlyObjCClass;
void consumeForwardImplicitlyObjCClass(ForwardImplicitlyObjCClass *_Nonnull obj);

@class ForwardExplicitlyObjCClass;
void consumeForwardExplicitlyObjCClass(ForwardExplicitlyObjCClass *_Nonnull obj);

@class ForwardHasSameCustomNameClass;
void consumeForwardHasSameCustomNameClass(ForwardHasSameCustomNameClass *_Nonnull obj);

@class ForwardNativeTypeHasDifferentCustomNameClass;
@class ObjCForwardNativeTypeHasDifferentCustomNameClass;
void consumeForwardNativeTypeHasDifferentCustomNameClass(ForwardNativeTypeHasDifferentCustomNameClass *_Nonnull obj);
void consumeObjCForwardNativeTypeHasDifferentCustomNameClass(ObjCForwardNativeTypeHasDifferentCustomNameClass *_Nonnull obj);

@class ForwardNativeTypeIsNonObjCClass;
void consumeForwardNativeTypeIsNonObjCClass(ForwardNativeTypeIsNonObjCClass *_Nonnull obj);

@class ForwardNativeTypeIsUnambiguouslyNonObjCClass;
void consumeForwardNativeTypeIsUnambiguouslyNonObjCClass(ForwardNativeTypeIsUnambiguouslyNonObjCClass *_Nonnull obj);

LANGUAGE_CLASS("BOGUS")
@interface BogusClass
@end

# pragma clang attribute pop

@interface CodiraClass (Category)
- (void)categoryMethod:(struct PureClangType)arg;
@end
