/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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

static void
generate_2int (const Type *t, const char *gen_name)
{
    Member *m;

    fprintf (headerfile,
	     "unsigned %s2int(%s);\n",
	     gen_name, gen_name);

    fprintf (codefile,
	     "unsigned %s2int(%s f)\n"
	     "{\n"
	     "unsigned r = 0;\n",
	     gen_name, gen_name);

    ASN1_TAILQ_FOREACH(m, t->members, members) {
	fprintf (codefile, "if(f.%s) r |= (1U << %d);\n",
		 m->gen_name, m->val);
    }
    fprintf (codefile, "return r;\n"
	     "}\n\n");
}

static void
generate_int2 (const Type *t, const char *gen_name)
{
    Member *m;

    fprintf (headerfile,
	     "%s int2%s(unsigned);\n",
	     gen_name, gen_name);

    fprintf (codefile,
	     "%s int2%s(unsigned n)\n"
	     "{\n"
	     "\t%s flags;\n\n"
	     "\tmemset(&flags, 0, sizeof(flags));\n\n",
	     gen_name, gen_name, gen_name);

    if(t->members) {
	ASN1_TAILQ_FOREACH(m, t->members, members) {
	    fprintf (codefile, "\tflags.%s = (n >> %d) & 1;\n",
		     m->gen_name, m->val);
	}
    }
    fprintf (codefile, "\treturn flags;\n"
	     "}\n\n");
}

/*
 * This depends on the bit string being declared in increasing order
 */

static void
generate_units (const Type *t, const char *gen_name)
{
    Member *m;

    if (template_flag) {
	fprintf (headerfile,
		 "extern const struct units *asn1_%s_table_units;\n",
		 gen_name);
	fprintf (headerfile, "#define asn1_%s_units() (asn1_%s_table_units)\n",
		 gen_name, gen_name);
    } else {
	fprintf (headerfile,
		 "const struct units * asn1_%s_units(void);\n",
		 gen_name);
    }

    fprintf (codefile,
	     "static struct units %s_units[] = {\n",
	     gen_name);

    if(t->members) {
	ASN1_TAILQ_FOREACH_REVERSE(m, t->members, memhead, members) {
	    fprintf (codefile,
		     "\t{\"%s\",\t1U << %d},\n", m->name, m->val);
	}
    }

    fprintf (codefile,
	     "\t{NULL,\t0}\n"
	     "};\n\n");

    if (template_flag)
	fprintf (codefile,
		 "const struct units * asn1_%s_table_units = %s_units;\n",
		 gen_name, gen_name);
    else
	fprintf (codefile,
		 "const struct units * asn1_%s_units(void){\n"
		 "return %s_units;\n"
		 "}\n\n",
		 gen_name, gen_name);


}

void
generate_glue (const Type *t, const char *gen_name)
{
    switch(t->type) {
    case TTag:
	generate_glue(t->subtype, gen_name);
	break;
    case TBitString :
	if (!ASN1_TAILQ_EMPTY(t->members)) {
	    generate_2int (t, gen_name);
	    generate_int2 (t, gen_name);
	    if (parse_units_flag)
		generate_units (t, gen_name);
	}
	break;
    default :
	break;
    }
}
