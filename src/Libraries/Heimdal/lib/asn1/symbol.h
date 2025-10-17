/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
/* $Id$ */

#ifndef _SYMBOL_H
#define _SYMBOL_H

#include "asn1_queue.h"

enum typetype {
    TBitString,
    TBoolean,
    TChoice,
    TEnumerated,
    TGeneralString,
    TTeletexString,
    TGeneralizedTime,
    TIA5String,
    TInteger,
    TNull,
    TOID,
    TOctetString,
    TPrintableString,
    TSequence,
    TSequenceOf,
    TSet,
    TSetOf,
    TTag,
    TType,
    TUTCTime,
    TUTF8String,
    TBMPString,
    TUniversalString,
    TVisibleString
};

typedef enum typetype Typetype;

struct type;

struct value {
    enum { booleanvalue,
	   nullvalue,
	   integervalue,
	   stringvalue,
	   objectidentifiervalue
    } type;
    union {
	int booleanvalue;
	int integervalue;
	char *stringvalue;
	struct objid *objectidentifiervalue;
    } u;
};

struct member {
    char *name;
    char *gen_name;
    char *label;
    int val;
    int optional;
    int ellipsis;
    struct type *type;
    ASN1_TAILQ_ENTRY(member) members;
    struct value *defval;
};

typedef struct member Member;

ASN1_TAILQ_HEAD(memhead, member);

struct symbol;

struct tagtype {
    int tagclass;
    int tagvalue;
    enum { TE_IMPLICIT, TE_EXPLICIT } tagenv;
};

struct range {
    int min;
    int max;
};

enum ctype { CT_CONTENTS, CT_USER } ;

struct constraint_spec;

struct type {
    Typetype type;
    struct memhead *members;
    struct symbol *symbol;
    struct type *subtype;
    struct tagtype tag;
    struct range *range;
    struct constraint_spec *constraint;
    unsigned long id;
};

typedef struct type Type;

struct constraint_spec {
    enum ctype ctype;
    union {
	struct {
	    Type *type;
	    struct value *encoding;
	} content;
    } u;
};

struct objid {
    const char *label;
    int value;
    struct objid *next;
};

struct symbol {
    char *name;
    char *gen_name;
    enum { SUndefined, SValue, Stype } stype;
    struct value *value;
    Type *type;
    struct {
	unsigned used;
	unsigned external;
    } flags;
};

typedef struct symbol Symbol;

void initsym (void);
Symbol *addsym (char *);
void output_name (char *);
int checksymbols(void);
#endif
