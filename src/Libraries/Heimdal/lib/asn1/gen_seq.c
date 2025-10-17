/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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
#include "gen_locl.h"

RCSID("$Id$");

void
generate_type_seq (const Symbol *s)
{
    char *subname;
    Type *type;

    if (!seq_type(s->name))
	return;
    type = s->type;
    while(type->type == TTag)
	type = type->subtype;

    if (type->type != TSequenceOf && type->type != TSetOf) {
	fprintf(stderr, "%s not seq of %d\n", s->name, (int)type->type);
	return;
    }

    /*
     * Require the subtype to be a type so we can name it and use
     * copy_/free_
     */

    if (type->subtype->type != TType) {
	fprintf(stderr, "%s subtype is not a type, can't generate "
	       "sequence code for this case: %d\n",
		s->name, (int)type->subtype->type);
	exit(1);
    }

    subname = type->subtype->symbol->gen_name;

    fprintf (headerfile,
	     "ASN1EXP int   ASN1CALL add_%s  (%s *, const %s *);\n"
	     "ASN1EXP int   ASN1CALL remove_%s  (%s *, unsigned int);\n",
	     s->gen_name, s->gen_name, subname,
	     s->gen_name, s->gen_name);

    fprintf (codefile, "int ASN1CALL\n"
	     "add_%s(%s *data, const %s *element)\n"
	     "{\n",
	     s->gen_name, s->gen_name, subname);

    fprintf (codefile,
	     "int ret;\n"
	     "void *ptr;\n"
	     "size_t size = 0;\n"
	     "\n"
	     "if (os_add_and_mul_overflow(data->len, 1, sizeof(data->val[0]), &size)) return ASN1_OVERRUN;\n"
	     "ptr = realloc(data->val, size);\n"
	     "if (ptr == NULL) return ENOMEM;\n"
	     "data->val = ptr;\n\n"
	     "ret = copy_%s(element, &data->val[data->len]);\n"
	     "if (ret) return ret;\n"
	     "data->len++;\n"
	     "return 0;\n",
	     subname);

    fprintf (codefile, "}\n\n");

    fprintf (codefile, "int ASN1CALL\n"
	     "remove_%s(%s *data, unsigned int element)\n"
	     "{\n",
	     s->gen_name, s->gen_name);

    fprintf (codefile,
	     "void *ptr;\n"
	     "size_t size, move_size = 0;\n"
	     "\n"
	     "if (data->len == 0 || element >= data->len)\n"
	     "\treturn ASN1_OVERRUN;\n"
	     "free_%s(&data->val[element]);\n"
	     "data->len--;\n"
	     /* don't move if its the last element */
	     "if (element < data->len) {\n"
	     "if (os_mul_overflow(sizeof(data->val[0]), (data->len - element), &move_size)) return ASN1_OVERRUN;\n"
	     "\tmemmove(&data->val[element], &data->val[element + 1], move_size);\n"
	     "}\n"
	     /* resize but don't care about failures since it doesn't matter */
	     "if (os_mul_overflow(data->len, sizeof(data->val[0]), &size)) return ASN1_OVERRUN;\n"
	     "ptr = realloc(data->val, size);\n"
	     "if (ptr != NULL || data->len == 0) data->val = ptr;\n"
	     "return 0;\n",
	     subname);

    fprintf (codefile, "}\n\n");
}
