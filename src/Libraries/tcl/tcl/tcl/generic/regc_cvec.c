/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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
/*
 * Notes:
 * Only (selected) functions in _this_ file should treat chr* as non-constant.
 */

/*
 - newcvec - allocate a new cvec
 ^ static struct cvec *newcvec(int, int);
 */
static struct cvec *
newcvec(
    int nchrs,			/* to hold this many chrs... */
    int nranges)		/* ... and this many ranges... */
{
    size_t nc = (size_t)nchrs + (size_t)nranges*2;
    size_t n = sizeof(struct cvec) + nc*sizeof(chr);
    struct cvec *cv = (struct cvec *) MALLOC(n);

    if (cv == NULL) {
	return NULL;
    }
    cv->chrspace = nchrs;
    cv->chrs = (chr *)(((char *)cv)+sizeof(struct cvec));
    cv->ranges = cv->chrs + nchrs;
    cv->rangespace = nranges;
    return clearcvec(cv);
}

/*
 - clearcvec - clear a possibly-new cvec
 * Returns pointer as convenience.
 ^ static struct cvec *clearcvec(struct cvec *);
 */
static struct cvec *
clearcvec(
    struct cvec *cv)		/* character vector */
{
    assert(cv != NULL);
    cv->nchrs = 0;
    cv->nranges = 0;
    return cv;
}

/*
 - addchr - add a chr to a cvec
 ^ static VOID addchr(struct cvec *, pchr);
 */
static void
addchr(
    struct cvec *cv,		/* character vector */
    pchr c)			/* character to add */
{
    cv->chrs[cv->nchrs++] = (chr)c;
}

/*
 - addrange - add a range to a cvec
 ^ static VOID addrange(struct cvec *, pchr, pchr);
 */
static void
addrange(
    struct cvec *cv,		/* character vector */
    pchr from,			/* first character of range */
    pchr to)			/* last character of range */
{
    assert(cv->nranges < cv->rangespace);
    cv->ranges[cv->nranges*2] = (chr)from;
    cv->ranges[cv->nranges*2 + 1] = (chr)to;
    cv->nranges++;
}

/*
 - getcvec - get a cvec, remembering it as v->cv
 ^ static struct cvec *getcvec(struct vars *, int, int);
 */
static struct cvec *
getcvec(
    struct vars *v,		/* context */
    int nchrs,			/* to hold this many chrs... */
    int nranges)		/* ... and this many ranges... */
{
    if ((v->cv != NULL) && (nchrs <= v->cv->chrspace) &&
	    (nranges <= v->cv->rangespace)) {
	return clearcvec(v->cv);
    }

    if (v->cv != NULL) {
	freecvec(v->cv);
    }
    v->cv = newcvec(nchrs, nranges);
    if (v->cv == NULL) {
	ERR(REG_ESPACE);
    }

    return v->cv;
}

/*
 - freecvec - free a cvec
 ^ static VOID freecvec(struct cvec *);
 */
static void
freecvec(
    struct cvec *cv)		/* character vector */
{
    FREE(cv);
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
