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
#include "gen_locl.h"

RCSID("$Id$");

static void
length_primitive (const char *typename,
		  const char *name,
		  const char *variable)
{
    fprintf (codefile, "%s += der_length_%s(%s);\n", variable, typename, name);
}

/* XXX same as der_length_tag */
static size_t
length_tag(unsigned int tag)
{
    size_t len = 0;

    if(tag <= 30)
	return 1;
    while(tag) {
	tag /= 128;
	len++;
    }
    return len + 1;
}


static int
length_type (const char *name, const Type *t,
	     const char *variable, const char *tmpstr)
{
    switch (t->type) {
    case TType:
#if 0
	length_type (name, t->symbol->type);
#endif
	fprintf (codefile, "%s += length_%s(%s);\n",
		 variable, t->symbol->gen_name, name);
	break;
    case TInteger:
	if(t->members) {
	    fprintf(codefile,
		    "{\n"
		    "int enumint = *%s;\n", name);
	    length_primitive ("integer", "&enumint", variable);
	    fprintf(codefile, "}\n");
	} else if (t->range == NULL) {
	    length_primitive ("heim_integer", name, variable);
	} else if (t->range->min == INT_MIN && t->range->max == INT_MAX) {
	    length_primitive ("integer", name, variable);
	} else if (t->range->min == 0 && (unsigned int)t->range->max == UINT_MAX) {
	    length_primitive ("unsigned", name, variable);
	} else if (t->range->min == 0 && t->range->max == INT_MAX) {
	    length_primitive ("unsigned", name, variable);
	} else
	    errx(1, "%s: unsupported range %d -> %d",
		 name, t->range->min, t->range->max);

	break;
    case TBoolean:
	fprintf (codefile, "%s += 1;\n", variable);
	break;
    case TEnumerated :
	length_primitive ("enumerated", name, variable);
	break;
    case TOctetString:
	length_primitive ("octet_string", name, variable);
	break;
    case TBitString: {
	if (ASN1_TAILQ_EMPTY(t->members))
	    length_primitive("bit_string", name, variable);
	else {
	    if (!rfc1510_bitstring) {
		Member *m;
		int pos = ASN1_TAILQ_LAST(t->members, memhead)->val;

		fprintf(codefile,
			"do {\n");
		ASN1_TAILQ_FOREACH_REVERSE(m, t->members, memhead, members) {
		    while (m->val / 8 < pos / 8) {
			pos -= 8;
		    }
		    fprintf (codefile,
			     "if((%s)->%s) { %s += %d; break; }\n",
			     name, m->gen_name, variable, (pos + 8) / 8);
		}
		fprintf(codefile,
			"} while(0);\n");
		fprintf (codefile, "%s += 1;\n", variable);
	    } else {
		fprintf (codefile, "%s += 5;\n", variable);
	    }
	}
	break;
    }
    case TSet:
    case TSequence:
    case TChoice: {
	Member *m, *have_ellipsis = NULL;

	if (t->members == NULL)
	    break;

	if(t->type == TChoice) {
	    fprintf (codefile, "switch((%s)->element) {\n"
		     "case ASN1_CHOICE_INVALID: ret = ASN1_INVALID_CHOICE;\nbreak;\n",
		     name);
	}

	ASN1_TAILQ_FOREACH(m, t->members, members) {
	    char *s;

	    if (m->ellipsis) {
		have_ellipsis = m;
		continue;
	    }

	    if(t->type == TChoice)
		fprintf(codefile, "case %s:\n", m->label);

	    if (asprintf (&s, "%s(%s)->%s%s",
			  m->optional ? "" : "&", name,
			  t->type == TChoice ? "u." : "", m->gen_name) < 0 || s == NULL)
		errx(1, "malloc");
	    if (m->optional)
		fprintf (codefile, "if(%s)", s);
	    else if(m->defval)
		gen_compare_defval(s + 1, m->defval);
	    fprintf (codefile, "{\n"
		     "size_t %s_oldret = %s;\n"
		     "%s = 0;\n", tmpstr, variable, variable);
	    length_type (s, m->type, "ret", m->gen_name);
	    fprintf (codefile, "ret += %s_oldret;\n", tmpstr);
	    fprintf (codefile, "}\n");
	    free (s);
	    if(t->type == TChoice)
		fprintf(codefile, "break;\n");
	}
	if(t->type == TChoice) {
	    if (have_ellipsis)
		fprintf(codefile,
			"case %s:\n"
			"ret += (%s)->u.%s.length;\n"
			"break;\n",
			have_ellipsis->label,
			name,
			have_ellipsis->gen_name);
	    fprintf (codefile, "}\n"); /* switch */
	}
	break;
    }
    case TSetOf:
    case TSequenceOf: {
	char *n = NULL;
	char *sname = NULL;

	fprintf (codefile,
		 "{\n"
		 "size_t %s_oldret = %s;\n"
		 "unsigned int n_%s;\n"
		 "%s = 0;\n",
		 tmpstr, variable, tmpstr, variable);

	fprintf (codefile, "for(n_%s = (%s)->len; n_%s > 0; --n_%s){\n",
		 tmpstr, name, tmpstr, tmpstr);
	fprintf (codefile, "size_t %s_for_oldret = %s;\n"
		 "%s = 0;\n", tmpstr, variable, variable);
	if (asprintf (&n, "&(%s)->val[n_%s - 1]", name, tmpstr) < 0  || n == NULL)
	    errx(1, "malloc");
	if (asprintf (&sname, "%s_S_Of", tmpstr) < 0 || sname == NULL)
	    errx(1, "malloc");
	length_type(n, t->subtype, variable, sname);
	fprintf (codefile, "%s += %s_for_oldret;\n",
		 variable, tmpstr);
	fprintf (codefile, "}\n");

	fprintf (codefile,
		 "%s += %s_oldret;\n"
		 "}\n", variable, tmpstr);
	free(n);
	free(sname);
	break;
    }
    case TGeneralizedTime:
	length_primitive ("generalized_time", name, variable);
	break;
    case TGeneralString:
	length_primitive ("general_string", name, variable);
	break;
    case TTeletexString:
	length_primitive ("general_string", name, variable);
	break;
    case TUTCTime:
	length_primitive ("utctime", name, variable);
	break;
    case TUTF8String:
	length_primitive ("utf8string", name, variable);
	break;
    case TPrintableString:
	length_primitive ("printable_string", name, variable);
	break;
    case TIA5String:
	length_primitive ("ia5_string", name, variable);
	break;
    case TBMPString:
	length_primitive ("bmp_string", name, variable);
	break;
    case TUniversalString:
	length_primitive ("universal_string", name, variable);
	break;
    case TVisibleString:
	length_primitive ("visible_string", name, variable);
	break;
    case TNull:
	fprintf (codefile, "/* NULL */\n");
	break;
    case TTag:{
    	char *tname = NULL;
	if (asprintf(&tname, "%s_tag", tmpstr) < 0 || tname == NULL)
	    errx(1, "malloc");
	length_type (name, t->subtype, variable, tname);
	fprintf (codefile, "ret += %lu + der_length_len (ret);\n",
		 (unsigned long)length_tag(t->tag.tagvalue));
	free(tname);
	break;
    }
    case TOID:
	length_primitive ("oid", name, variable);
	break;
    }
    return 0;
}

void
generate_type_length (const Symbol *s)
{
    fprintf (codefile,
	     "size_t ASN1CALL\n"
	     "length_%s(const %s *data)\n"
	     "{\n"
	     "size_t ret = 0;\n",
	     s->gen_name, s->gen_name);

    length_type ("data", s->type, "ret", "Top");
    fprintf (codefile, "return ret;\n}\n\n");
}

