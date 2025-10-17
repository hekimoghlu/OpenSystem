/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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

// Structures to help us define fake classes in tests.

#include <objc/runtime.h>
#include <ptrauth.h>
#include <stdint.h>

struct ObjCClass {
    struct ObjCClass * __ptrauth_objc_isa_pointer isa;
    struct ObjCClass * __ptrauth_objc_super_pointer superclass;
    void *cachePtr;
    uintptr_t zero;
    const void *__ptrauth_objc_class_ro data;
};

struct ObjCClass_ro {
    uint32_t flags;
    uint32_t instanceStart;
    uint32_t instanceSize;
#ifdef __LP64__
    uint32_t reserved;
#endif

    union {
        const uint8_t * ivarLayout;
        struct ObjCClass * nonMetaClass;
    };

    const char * name;
    struct ObjCMethodList * __ptrauth_objc_method_list_pointer baseMethodList;
    struct protocol_list_t * baseProtocols;
    const struct ivar_list_t * ivars;

    const uint8_t * weakIvarLayout;
    struct property_list_t *baseProperties;
};

struct ObjCMethod {
    char *name;
    char *type;
    IMP imp;
};

struct ObjCMethodList {
    uint32_t sizeAndFlags;
    uint32_t count;
    struct ObjCMethod methods[];
};

struct ObjCMethodSmall {
    int32_t nameOffset;
    int32_t typeOffset;
    int32_t impOffset;
};

struct ObjCMethodListSmall {
    uint32_t sizeAndFlags;
    uint32_t count;
    struct ObjCMethodSmall methods[];
};

struct ObjCCategory {
    const char *name;
    struct ObjCClass *cls;
    struct ObjCMethodList * __ptrauth_objc_method_list_pointer instanceMethods;
    struct ObjCMethodList * __ptrauth_objc_method_list_pointer classMethods;
    struct protocol_list_t *protocols;
    struct property_list_t *instanceProperties;
    struct property_list_t *classProperties;
};

extern struct ObjCClass OBJC_METACLASS_$_NSObject;
extern struct ObjCClass OBJC_CLASS_$_NSObject;

#define RO_META (1<<0)

#if __LP64__
#define PTR ".quad "
#else
#define PTR ".long "
#endif

#define SELREF(name) \
    asm(".section __TEXT,__cstring\n" \
        "_" #name "Selector:\n" \
        ".asciz \"" #name "\"\n" \
        ".section __DATA,__objc_selrefs,literal_pointers,no_dead_strip\n" \
        "_" #name "Ref:\n" \
        PTR "_" #name "Selector\n" \
    );

#define SMALL_METHOD_LIST(name, count, methods) \
    extern struct ObjCMethodListSmall name; \
    asm(".text\n" \
        ".p2align 2\n" \
        "_" #name ":\n" \
        " .long 12 | 0x80000000" \
        "\n .long " #count "\n" \
        methods \
    );

#define SMALL_METHOD(name, types, imp) \
    ".long _" #name "Ref - .\n" \
    ".long _" #types " - .\n" \
    ".long _" #imp " - .\n"
