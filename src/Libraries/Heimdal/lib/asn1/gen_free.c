/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
free_primitive (const char *typename, const char *name)
{
    fprintf (codefile, "der_free_%s(%s);\n", typename, name);
}

static void
free_type (const char *name, const Type *t, int preserve)
{
    switch (t->type) {
    case TType:
#if 0
	free_type (name, t->symbol->type, preserve);
#endif
	fprintf (codefile, "free_%s(%s);\n", t->symbol->gen_name, name);
	break;
    case TInteger:
	if (t->range == NULL && t->members == NULL) {
	    free_primitive ("heim_integer", name);
	    break;
	}
    case TBoolean:
    case TEnumerated :
    case TNull:
    case TGeneralizedTime:
    case TUTCTime:
        /*
         * This doesn't do much, but it leaves zeros where garbage might
         * otherwise have been found.  Gets us closer to having the equivalent
         * of a memset()-to-zero data structure after calling the free
         * functions.
         */
        fprintf(codefile, "*%s = 0;\n", name);
	break;
    case TBitString:
	if (ASN1_TAILQ_EMPTY(t->members))
	    free_primitive("bit_string", name);
	break;
    case TOctetString:
	free_primitive ("octet_string", name);
	break;
    case TChoice:
    case TSet:
    case TSequence: {
	Member *m, *have_ellipsis = NULL;

	if (t->members == NULL)
	    break;

	if ((t->type == TSequence || t->type == TChoice) && preserve)
	    fprintf(codefile, "der_free_octet_string(&data->_save);\n");

	if(t->type == TChoice) {
	    fprintf(codefile, "switch((%s)->element) {\n", name);
	    fprintf(codefile,
		    "case ASN1_CHOICE_INVALID: break;\n");
	}

	ASN1_TAILQ_FOREACH(m, t->members, members) {
	    char *s;

	    if (m->ellipsis){
		have_ellipsis = m;
		continue;
	    }

	    if(t->type == TChoice)
		fprintf(codefile, "case %s:\n", m->label);
	    if (asprintf (&s, "%s(%s)->%s%s",
			  m->optional ? "" : "&", name,
			  t->type == TChoice ? "u." : "", m->gen_name) < 0 || s == NULL)
		errx(1, "malloc");
	    if(m->optional)
		fprintf(codefile, "if(%s) {\n", s);
	    free_type (s, m->type, FALSE);
	    if(m->optional)
		fprintf(codefile,
			"free(%s);\n"
			"%s = NULL;\n"
			"}\n",s, s);
	    free (s);
	    if(t->type == TChoice)
		fprintf(codefile, "break;\n");
	}

	if(t->type == TChoice) {
	    if (have_ellipsis)
		fprintf(codefile,
			"case %s:\n"
			"der_free_octet_string(&(%s)->u.%s);\n"
			"break;",
			have_ellipsis->label,
			name, have_ellipsis->gen_name);
	    fprintf(codefile, "}\n");
	}
	break;
    }
    case TSetOf:
    case TSequenceOf: {
	char *n;

	fprintf (codefile, "while((%s)->len){\n", name);
	if (asprintf (&n, "&(%s)->val[(%s)->len-1]", name, name) < 0 || n == NULL)
	    errx(1, "malloc");
	free_type(n, t->subtype, FALSE);
	fprintf(codefile,
		"(%s)->len--;\n"
		"}\n",
		name);
	fprintf(codefile,
		"free((%s)->val);\n"
		"(%s)->val = NULL;\n", name, name);
	free(n);
	break;
    }
    case TGeneralString:
	free_primitive ("general_string", name);
	break;
    case TTeletexString:
	free_primitive ("general_string", name);
	break;
    case TUTF8String:
	free_primitive ("utf8string", name);
	break;
    case TPrintableString:
	free_primitive ("printable_string", name);
	break;
    case TIA5String:
	free_primitive ("ia5_string", name);
	break;
    case TBMPString:
	free_primitive ("bmp_string", name);
	break;
    case TUniversalString:
	free_primitive ("universal_string", name);
	break;
    case TVisibleString:
	free_primitive ("visible_string", name);
	break;
    case TTag:
	free_type (name, t->subtype, preserve);
	break;
    case TOID :
	free_primitive ("oid", name);
	break;
    }
}

void
generate_type_free (const Symbol *s)
{
    int preserve = preserve_type(s->name) ? TRUE : FALSE;

    fprintf (codefile, "void ASN1CALL\n"
	     "free_%s(%s * HEIMDAL_UNUSED_ATTRIBUTE data)\n"
	     "{\n",
	     s->gen_name, s->gen_name);

    free_type ("data", s->type, preserve);
    fprintf (codefile, "}\n\n");
}

