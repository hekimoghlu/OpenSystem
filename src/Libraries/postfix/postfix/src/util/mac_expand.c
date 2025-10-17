/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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
/* System library. */

#include <sys_defs.h>
#include <ctype.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <mymalloc.h>
#include <stringops.h>
#include <name_code.h>
#include <mac_parse.h>
#include <mac_expand.h>

 /*
  * Little helper structure.
  */
typedef struct {
    VSTRING *result;			/* result buffer */
    int     flags;			/* features */
    const char *filter;			/* character filter */
    MAC_EXP_LOOKUP_FN lookup;		/* lookup routine */
    void   *context;			/* caller context */
    int     status;			/* findings */
    int     level;			/* nesting level */
} MAC_EXP_CONTEXT;

 /*
  * Support for relational expressions.
  * 
  * As of Postfix 2.2, ${attr-name?result} or ${attr-name:result} return the
  * result respectively when the parameter value is non-empty, or when the
  * parameter value is undefined or empty; support for the ternary ?:
  * operator was anticipated, but not implemented for 10 years.
  * 
  * To make ${relational-expr?result} and ${relational-expr:result} work as
  * expected without breaking the way that ? and : work, relational
  * expressions evaluate to a non-empty or empty value. It does not matter
  * what non-empty value we use for TRUE. However we must not use the
  * undefined (null pointer) value for FALSE - that would raise the
  * MAC_PARSE_UNDEF flag.
  * 
  * The value of a relational expression can be exposed with ${relational-expr},
  * i.e. a relational expression that is not followed by ? or : conditional
  * expansion.
  */
#define MAC_EXP_BVAL_TRUE	"true"
#define MAC_EXP_BVAL_FALSE	""

 /*
  * Relational operators.
  */
#define MAC_EXP_OP_STR_EQ	"=="
#define MAC_EXP_OP_STR_NE	"!="
#define MAC_EXP_OP_STR_LT	"<"
#define MAC_EXP_OP_STR_LE	"<="
#define MAC_EXP_OP_STR_GE	">="
#define MAC_EXP_OP_STR_GT	">"
#define MAC_EXP_OP_STR_ANY	"\"" MAC_EXP_OP_STR_EQ \
				"\" or \"" MAC_EXP_OP_STR_NE "\"" \
				"\" or \"" MAC_EXP_OP_STR_LT "\"" \
				"\" or \"" MAC_EXP_OP_STR_LE "\"" \
				"\" or \"" MAC_EXP_OP_STR_GE "\"" \
				"\" or \"" MAC_EXP_OP_STR_GT "\""

#define MAC_EXP_OP_TOK_NONE	0
#define MAC_EXP_OP_TOK_EQ	1
#define MAC_EXP_OP_TOK_NE	2
#define MAC_EXP_OP_TOK_LT	3
#define MAC_EXP_OP_TOK_LE	4
#define MAC_EXP_OP_TOK_GE	5
#define MAC_EXP_OP_TOK_GT	6

static const NAME_CODE mac_exp_op_table[] =
{
    MAC_EXP_OP_STR_EQ, MAC_EXP_OP_TOK_EQ,
    MAC_EXP_OP_STR_NE, MAC_EXP_OP_TOK_NE,
    MAC_EXP_OP_STR_LT, MAC_EXP_OP_TOK_LT,
    MAC_EXP_OP_STR_LE, MAC_EXP_OP_TOK_LE,
    MAC_EXP_OP_STR_GE, MAC_EXP_OP_TOK_GE,
    MAC_EXP_OP_STR_GT, MAC_EXP_OP_TOK_GT,
    0, MAC_EXP_OP_TOK_NONE,
};

 /*
  * The whitespace separator set.
  */
#define MAC_EXP_WHITESPACE	CHARS_SPACE

/* atol_or_die - convert or die */

static long atol_or_die(const char *strval)
{
    long    result;
    char   *remainder;

    result = strtol(strval, &remainder, 10);
    if (*strval == 0 /* can't happen */ || *remainder != 0 || errno == ERANGE)
	msg_fatal("mac_exp_eval: bad conversion: %s", strval);
    return (result);
}

/* mac_exp_eval - evaluate binary expression */

static int mac_exp_eval(const char *left, int tok_val,
			        const char *rite)
{
    static const char myname[] = "mac_exp_eval";
    long    delta;

    /*
     * Numerical or string comparison.
     */
    if (alldig(left) && alldig(rite)) {
	delta = atol_or_die(left) - atol_or_die(rite);
    } else {
	delta = strcmp(left, rite);
    }
    switch (tok_val) {
    case MAC_EXP_OP_TOK_EQ:
	return (delta == 0);
    case MAC_EXP_OP_TOK_NE:
	return (delta != 0);
    case MAC_EXP_OP_TOK_LT:
	return (delta < 0);
    case MAC_EXP_OP_TOK_LE:
	return (delta <= 0);
    case MAC_EXP_OP_TOK_GE:
	return (delta >= 0);
    case MAC_EXP_OP_TOK_GT:
	return (delta > 0);
    default:
	msg_panic("%s: unknown operator: %d",
		  myname, tok_val);
    }
}

/* mac_exp_parse_error - report parse error, set error flag, return status */

static int PRINTFLIKE(2, 3) mac_exp_parse_error(MAC_EXP_CONTEXT *mc,
						        const char *fmt,...)
{
    va_list ap;

    va_start(ap, fmt);
    vmsg_warn(fmt, ap);
    va_end(ap);
    return (mc->status |= MAC_PARSE_ERROR);
};

/* MAC_EXP_ERR_RETURN - report parse error, set error flag, return status */

#define MAC_EXP_ERR_RETURN(mc, fmt, ...) do { \
	return (mac_exp_parse_error(mc, fmt, __VA_ARGS__)); \
    } while (0)

 /*
  * Postfix 3.0 introduces support for {text} operands. Only with these do we
  * support the ternary ?: operator and relational operators.
  * 
  * We cannot support operators in random text, because that would break Postfix
  * 2.11 compatibility. For example, with the expression "${name?value}", the
  * value is random text that may contain ':', '?', '{' and '}' characters.
  * In particular, with Postfix 2.2 .. 2.11, "${name??foo:{b}ar}" evaluates
  * to "?foo:{b}ar" or empty. There are explicit tests in this directory and
  * the postconf directory to ensure that Postfix 2.11 compatibility is
  * maintained.
  * 
  * Ideally, future Postfix configurations enclose random text operands inside
  * {} braces. These allow whitespace around operands, which improves
  * readability.
  */

/* MAC_EXP_FIND_LEFT_CURLY - skip over whitespace to '{', advance read ptr */

#define MAC_EXP_FIND_LEFT_CURLY(len, cp) \
	((cp[len = strspn(cp, MAC_EXP_WHITESPACE)] == '{') ? \
	 (cp += len) : 0)

/* mac_exp_extract_curly_payload - balance {}, skip whitespace, return payload */

static char *mac_exp_extract_curly_payload(MAC_EXP_CONTEXT *mc, char **bp)
{
    char   *payload;
    char   *cp;
    int     level;
    int     ch;

    /*
     * Extract the payload and balance the {}. The caller is expected to skip
     * leading whitespace before the {. See MAC_EXP_FIND_LEFT_CURLY().
     */
    for (level = 1, cp = *bp, payload = ++cp; /* see below */ ; cp++) {
	if ((ch = *cp) == 0) {
	    mac_exp_parse_error(mc, "unbalanced {} in attribute expression: "
				"\"%s\"",
				*bp);
	    return (0);
	} else if (ch == '{') {
	    level++;
	} else if (ch == '}') {
	    if (--level <= 0)
		break;
	}
    }
    *cp++ = 0;

    /*
     * Skip trailing whitespace after }.
     */
    *bp = cp + strspn(cp, MAC_EXP_WHITESPACE);
    return (payload);
}

/* mac_exp_parse_relational - parse relational expression, advance read ptr */

static int mac_exp_parse_relational(MAC_EXP_CONTEXT *mc, const char **lookup,
				            char **bp)
{
    char   *cp = *bp;
    VSTRING *left_op_buf;
    VSTRING *rite_op_buf;
    const char *left_op_strval;
    const char *rite_op_strval;
    char   *op_pos;
    char   *op_strval;
    size_t  op_len;
    int     op_tokval;
    int     op_result;
    size_t  tmp_len;

    /*
     * Left operand. The caller is expected to skip leading whitespace before
     * the {. See MAC_EXP_FIND_LEFT_CURLY().
     */
    if ((left_op_strval = mac_exp_extract_curly_payload(mc, &cp)) == 0)
	return (mc->status);

    /*
     * Operator. Todo: regexp operator.
     */
    op_pos = cp;
    op_len = strspn(cp, "<>!=?+-*/~&|%");	/* for better diagnostics. */
    op_strval = mystrndup(cp, op_len);
    op_tokval = name_code(mac_exp_op_table, NAME_CODE_FLAG_NONE, op_strval);
    myfree(op_strval);
    if (op_tokval == MAC_EXP_OP_TOK_NONE)
	MAC_EXP_ERR_RETURN(mc, "%s expected at: \"...%s}>>>%.20s\"",
			   MAC_EXP_OP_STR_ANY, left_op_strval, cp);
    cp += op_len;

    /*
     * Right operand. Todo: syntax may depend on operator.
     */
    if (MAC_EXP_FIND_LEFT_CURLY(tmp_len, cp) == 0)
	MAC_EXP_ERR_RETURN(mc, "\"{expression}\" expected at: "
			   "\"...{%s} %.*s>>>%.20s\"",
			   left_op_strval, (int) op_len, op_pos, cp);
    if ((rite_op_strval = mac_exp_extract_curly_payload(mc, &cp)) == 0)
	return (mc->status);

    /*
     * Evaluate the relational expression. Todo: regexp support.
     */
    mc->status |=
	mac_expand(left_op_buf = vstring_alloc(100), left_op_strval,
		   mc->flags, mc->filter, mc->lookup, mc->context);
    mc->status |=
	mac_expand(rite_op_buf = vstring_alloc(100), rite_op_strval,
		   mc->flags, mc->filter, mc->lookup, mc->context);
    op_result = mac_exp_eval(vstring_str(left_op_buf), op_tokval,
			     vstring_str(rite_op_buf));
    vstring_free(left_op_buf);
    vstring_free(rite_op_buf);
    if (mc->status & MAC_PARSE_ERROR)
	return (mc->status);

    /*
     * Here, we fake up a non-empty or empty parameter value lookup result,
     * for compatibility with the historical code that looks named parameter
     * values.
     */
    *lookup = (op_result ? MAC_EXP_BVAL_TRUE : MAC_EXP_BVAL_FALSE);
    *bp = cp;
    return (0);
}

/* mac_expand_callback - callback for mac_parse */

static int mac_expand_callback(int type, VSTRING *buf, void *ptr)
{
    static const char myname[] = "mac_expand_callback";
    MAC_EXP_CONTEXT *mc = (MAC_EXP_CONTEXT *) ptr;
    int     lookup_mode;
    const char *lookup;
    char   *cp;
    int     ch;
    ssize_t res_len;
    ssize_t tmp_len;
    const char *res_iftrue;
    const char *res_iffalse;

    /*
     * Sanity check.
     */
    if (mc->level++ > 100)
	mac_exp_parse_error(mc, "unreasonable macro call nesting: \"%s\"",
			    vstring_str(buf));
    if (mc->status & MAC_PARSE_ERROR)
	return (mc->status);

    /*
     * Named parameter or relational expression. In case of a syntax error,
     * return without doing damage, and issue a warning instead.
     */
    if (type == MAC_PARSE_EXPR) {

	cp = vstring_str(buf);

	/*
	 * Relational expression. If recursion is disabled, perform only one
	 * level of $name expansion.
	 */
	if (MAC_EXP_FIND_LEFT_CURLY(tmp_len, cp)) {
	    if (mac_exp_parse_relational(mc, &lookup, &cp) != 0)
		return (mc->status);

	    /*
	     * Look for the ? or : operator.
	     */
	    if ((ch = *cp) != 0) {
		if (ch != '?' && ch != ':')
		    MAC_EXP_ERR_RETURN(mc, "\"?\" or \":\" expected at: "
				       "\"...}>>>%.20s\"", cp);
		cp++;
	    }
	}

	/*
	 * Named parameter.
	 */
	else {

	    /*
	     * Look for the ? or : operator. In case of a syntax error,
	     * return without doing damage, and issue a warning instead.
	     */
	    for ( /* void */ ; /* void */ ; cp++) {
		if ((ch = *cp) == 0) {
		    lookup_mode = MAC_EXP_MODE_USE;
		    break;
		}
		if (ch == '?' || ch == ':') {
		    *cp++ = 0;
		    lookup_mode = MAC_EXP_MODE_TEST;
		    break;
		}
		if (!ISALNUM(ch) && ch != '_') {
		    MAC_EXP_ERR_RETURN(mc, "attribute name syntax error at: "
				       "\"...%.*s>>>%.20s\"",
				       (int) (cp - vstring_str(buf)),
				       vstring_str(buf), cp);
		}
	    }

	    /*
	     * Look up the named parameter. Todo: allow the lookup function
	     * to specify if the result is safe for $name expanson.
	     */
	    lookup = mc->lookup(vstring_str(buf), lookup_mode, mc->context);
	}

	/*
	 * Return the requested result. After parsing the result operand
	 * following ?, we fall through to parse the result operand following
	 * :. This is necessary with the ternary ?: operator: first, with
	 * MAC_EXP_FLAG_SCAN to parse both result operands with mac_parse(),
	 * and second, to find garbage after any result operand. Without
	 * MAC_EXP_FLAG_SCAN the content of only one of the ?: result
	 * operands will be parsed with mac_parse(); syntax errors in the
	 * other operand will be missed.
	 */
	switch (ch) {
	case '?':
	    if (MAC_EXP_FIND_LEFT_CURLY(tmp_len, cp)) {
		if ((res_iftrue = mac_exp_extract_curly_payload(mc, &cp)) == 0)
		    return (mc->status);
	    } else {
		res_iftrue = cp;
		cp = "";			/* no left-over text */
	    }
	    if ((lookup != 0 && *lookup != 0) || (mc->flags & MAC_EXP_FLAG_SCAN))
		mc->status |= mac_parse(res_iftrue, mac_expand_callback,
					(void *) mc);
	    if (*cp == 0)			/* end of input, OK */
		break;
	    if (*cp != ':')			/* garbage */
		MAC_EXP_ERR_RETURN(mc, "\":\" expected at: "
				   "\"...%s}>>>%.20s\"", res_iftrue, cp);
	    cp += 1;
	    /* FALLTHROUGH: do not remove, see comment above. */
	case ':':
	    if (MAC_EXP_FIND_LEFT_CURLY(tmp_len, cp)) {
		if ((res_iffalse = mac_exp_extract_curly_payload(mc, &cp)) == 0)
		    return (mc->status);
	    } else {
		res_iffalse = cp;
		cp = "";			/* no left-over text */
	    }
	    if (lookup == 0 || *lookup == 0 || (mc->flags & MAC_EXP_FLAG_SCAN))
		mc->status |= mac_parse(res_iffalse, mac_expand_callback,
					(void *) mc);
	    if (*cp != 0)			/* garbage */
		MAC_EXP_ERR_RETURN(mc, "unexpected input at: "
				   "\"...%s}>>>%.20s\"", res_iffalse, cp);
	    break;
	case 0:
	    if (lookup == 0) {
		mc->status |= MAC_PARSE_UNDEF;
	    } else if (*lookup == 0 || (mc->flags & MAC_EXP_FLAG_SCAN)) {
		 /* void */ ;
	    } else if (mc->flags & MAC_EXP_FLAG_RECURSE) {
		vstring_strcpy(buf, lookup);
		mc->status |= mac_parse(vstring_str(buf), mac_expand_callback,
					(void *) mc);
	    } else {
		res_len = VSTRING_LEN(mc->result);
		vstring_strcat(mc->result, lookup);
		if (mc->flags & MAC_EXP_FLAG_PRINTABLE) {
		    printable(vstring_str(mc->result) + res_len, '_');
		} else if (mc->filter) {
		    cp = vstring_str(mc->result) + res_len;
		    while (*(cp += strspn(cp, mc->filter)))
			*cp++ = '_';
		}
	    }
	    break;
	default:
	    msg_panic("%s: unknown operator code %d", myname, ch);
	}
    }

    /*
     * Literal text.
     */
    else if ((mc->flags & MAC_EXP_FLAG_SCAN) == 0) {
	vstring_strcat(mc->result, vstring_str(buf));
    }
    mc->level--;

    return (mc->status);
}

/* mac_expand - expand $name instances */

int     mac_expand(VSTRING *result, const char *pattern, int flags,
		           const char *filter,
		           MAC_EXP_LOOKUP_FN lookup, void *context)
{
    MAC_EXP_CONTEXT mc;
    int     status;

    /*
     * Bundle up the request and do the substitutions.
     */
    mc.result = result;
    mc.flags = flags;
    mc.filter = filter;
    mc.lookup = lookup;
    mc.context = context;
    mc.status = 0;
    mc.level = 0;
    if ((flags & (MAC_EXP_FLAG_APPEND | MAC_EXP_FLAG_SCAN)) == 0)
	VSTRING_RESET(result);
    status = mac_parse(pattern, mac_expand_callback, (void *) &mc);
    if ((flags & MAC_EXP_FLAG_SCAN) == 0)
	VSTRING_TERMINATE(result);

    return (status);
}

#ifdef TEST

 /*
  * This code certainly deserves a stand-alone test program.
  */
#include <stdlib.h>
#include <stringops.h>
#include <htable.h>
#include <vstream.h>
#include <vstring_vstream.h>

static const char *lookup(const char *name, int unused_mode, void *context)
{
    HTABLE *table = (HTABLE *) context;

    return (htable_find(table, name));
}

int     main(int unused_argc, char **unused_argv)
{
    VSTRING *buf = vstring_alloc(100);
    VSTRING *result = vstring_alloc(100);
    char   *cp;
    char   *name;
    char   *value;
    HTABLE *table;
    int     stat;

    while (!vstream_feof(VSTREAM_IN)) {

	table = htable_create(0);

	/*
	 * Read a block of definitions, terminated with an empty line.
	 */
	while (vstring_get_nonl(buf, VSTREAM_IN) != VSTREAM_EOF) {
	    vstream_printf("<< %s\n", vstring_str(buf));
	    vstream_fflush(VSTREAM_OUT);
	    if (VSTRING_LEN(buf) == 0)
		break;
	    cp = vstring_str(buf);
	    name = mystrtok(&cp, CHARS_SPACE "=");
	    value = mystrtok(&cp, CHARS_SPACE "=");
	    htable_enter(table, name, value ? mystrdup(value) : 0);
	}

	/*
	 * Read a block of patterns, terminated with an empty line or EOF.
	 */
	while (vstring_get_nonl(buf, VSTREAM_IN) != VSTREAM_EOF) {
	    vstream_printf("<< %s\n", vstring_str(buf));
	    vstream_fflush(VSTREAM_OUT);
	    if (VSTRING_LEN(buf) == 0)
		break;
	    cp = vstring_str(buf);
	    VSTRING_RESET(result);
	    stat = mac_expand(result, vstring_str(buf), MAC_EXP_FLAG_NONE,
			      (char *) 0, lookup, (void *) table);
	    vstream_printf("stat=%d result=%s\n", stat, vstring_str(result));
	    vstream_fflush(VSTREAM_OUT);
	}
	htable_free(table, myfree);
	vstream_printf("\n");
    }

    /*
     * Clean up.
     */
    vstring_free(buf);
    vstring_free(result);
    exit(0);
}

#endif
