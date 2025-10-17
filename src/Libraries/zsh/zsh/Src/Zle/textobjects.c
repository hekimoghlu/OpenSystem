/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
#include "zle.mdh"
#include "textobjects.pro"

static int
blankwordclass(ZLE_CHAR_T x)
{
    return (ZC_iblank(x) ? 0 : 1);
}

/**/
int
selectword(UNUSED(char **args))
{
    int n = zmult;
    int all = IS_THINGY(bindk, selectaword) ||
	IS_THINGY(bindk, selectablankword);
    int (*viclass)(ZLE_CHAR_T) = (IS_THINGY(bindk, selectaword) ||
	    IS_THINGY(bindk, selectinword)) ? wordclass : blankwordclass;
    int sclass = viclass(zleline[zlecs]);
    int doblanks = all && sclass;

    if (!region_active || zlecs == mark || mark == -1) {
	/* search back to first character of same class as the start position
	 * also stop at the beginning of the line */
	mark = zlecs;
	while (mark) {
	    int pos = mark;
	    DECPOS(pos);
	    if (zleline[pos] == ZWC('\n') || viclass(zleline[pos]) != sclass)
		break;
	    mark = pos;
	}
	/* similarly scan forward over characters of the same class */
	while (zlecs < zlell) {
	    INCCS();
	    int pos = zlecs;
	    /* single newlines within blanks are included */
	    if (all && !sclass && pos < zlell && zleline[pos] == ZWC('\n'))
		INCPOS(pos);

	    if (zleline[pos] == ZWC('\n') || viclass(zleline[pos]) != sclass)
		break;
	}

	if (all) {
	    int nclass = viclass(zleline[zlecs]);
	    /* if either start or new position is blank advance over
	     * a new block of characters of a common type */
	    if (!nclass || !sclass) {
		while (zlecs < zlell) {
		    INCCS();
		    if (zleline[zlecs] == ZWC('\n') ||
			    viclass(zleline[zlecs]) != nclass)
			break;
		}
		if (n < 2)
		    doblanks = 0;
	    }
	}
    } else {
	/* For visual mode, advance one char so repeated
	 * invocations select subsequent words */
	if (zlecs > mark) {
	    if (zlecs < zlell)
		INCCS();
	} else if (zlecs)
	    DECCS();
	if (zlecs < mark) {
	    /* visual mode with the cursor before the mark: move cursor back */
	    while (n-- > 0) {
		int pos = zlecs;
		/* first over blanks */
		if (all && (!viclass(zleline[pos]) ||
			zleline[pos] == ZWC('\n'))) {
		    all = 0;
		    while (pos) {
			DECPOS(pos);
			if (zleline[pos] == ZWC('\n'))
			    break;
			zlecs = pos;
			if (viclass(zleline[pos]))
			    break;
		    }
		} else if (zlecs && zleline[zlecs] == ZWC('\n')) {
		    /* for in widgets pass over one newline */
		    DECPOS(pos);
		    if (zleline[pos] != ZWC('\n'))
			zlecs = pos;
		}
		pos = zlecs;
		sclass = viclass(zleline[zlecs]);
		/* now retreat over non-blanks */
		while (zleline[pos] != ZWC('\n') &&
			viclass(zleline[pos]) == sclass) {
		    zlecs = pos;
		    if (!pos) {
			zlecs = 0;
			break;
		    }
		    DECPOS(pos);
		}
		/* blanks again but only if there were none first time */
		if (all && zlecs) {
		    pos = zlecs;
		    DECPOS(pos);
		    if (!viclass(zleline[pos])) {
			while (pos) {
			    DECPOS(pos);
			    if (zleline[pos] == ZWC('\n') ||
				    viclass(zleline[pos]))
				break;
			    zlecs = pos;
			}
		    }
		}
	    }
	    return 0;
	}
	n++;
	doblanks = 0;
    }
    region_active = !!region_active; /* force to character wise */

    /* for each digit argument, advance over further block of one class */
    while (--n > 0) {
	if (zlecs < zlell && zleline[zlecs] == ZWC('\n'))
	    INCCS();
	sclass = viclass(zleline[zlecs]);
	while (zlecs < zlell) {
	    INCCS();
	    if (zleline[zlecs] == ZWC('\n') ||
		    viclass(zleline[zlecs]) != sclass)
		break;
	}
	/* for 'a' widgets, advance extra block if either consists of blanks */
	if (all) {
	    if (zlecs < zlell && zleline[zlecs] == ZWC('\n'))
		INCCS();
	    if (!sclass || !viclass(zleline[zlecs]) ) {
		sclass = viclass(zleline[zlecs]);
		if (n == 1 && !sclass)
		    doblanks = 0;
		while (zlecs < zlell) {
		    INCCS();
		    if (zleline[zlecs] == ZWC('\n') ||
			    viclass(zleline[zlecs]) != sclass)
			break;
		}
	    }
	}
    }

    /* if we didn't remove blanks at either end we remove some at the start */
    if (doblanks) {
	int pos = mark;
	while (pos) {
	    DECPOS(pos);
	    /* don't remove blanks at the start of the line, i.e indentation */
	    if (zleline[pos] == ZWC('\n'))
		break;
	    if (!ZC_iblank(zleline[pos])) {
		INCPOS(pos);
		mark = pos;
		break;
	    }
	}
    }
    /* Adjustment: vi operators don't include the cursor position, in insert
     * or emacs mode the region also doesn't but for vi visual mode it is
     * included. */
    if (!virangeflag) {
	if (!invicmdmode())
	    region_active = 1;
	else if (zlecs && zlecs > mark)
	    DECCS();
    }

    return 0;
}

/**/
int
selectargument(UNUSED(char **args))
{
    int ne = noerrs, ocs = zlemetacs;
    int owb = wb, owe= we, oadx = addedx, ona = noaliases;
    char *p;
    int ll, cs;
    char *linein;
    int wend = 0, wcur = 0;
    int n = zmult;
    int *wstarts;
    int tmpsz;

    if (n < 1 || 2*n > zlell + 1)
	return 1;

    /* if used from emacs mode enable the region */
    if (!invicmdmode()) {
	region_active = 1;
	mark = zlecs;
    }

    wstarts = (int *) zhalloc(n * sizeof(int));
    memset(wstarts, 0, n * sizeof(int));

    addedx = 0;
    noerrs = 1;
    zcontext_save();
    lexflags = LEXFLAGS_ACTIVE;
    linein = zlegetline(&ll, &cs);
    zlemetall = ll;
    zlemetacs = cs;

    if (!isfirstln && chline) {
       p = (char *) zhalloc(hptr - chline + zlemetall + 2);
       memcpy(p, chline, hptr - chline);
       memcpy(p + (hptr - chline), linein, ll);
       p[(hptr - chline) + ll] = '\0';
       inpush(p, 0, NULL);
       zlemetacs += hptr - chline;
    } else {
       p = (char *) zhalloc(ll + 1);
       memcpy(p, linein, ll);
       p[ll] = '\0';
       inpush(p, 0, NULL);
    }
    if (zlemetacs)
       zlemetacs--;
    strinbeg(0);
    noaliases = 1;
    do {
       wstarts[wcur++] = wend;
       wcur %= n;
       ctxtlex();
       if (tok == ENDINPUT || tok == LEXERR)
           break;
       wend = zlemetall - inbufct;
    } while (tok != ENDINPUT && tok != LEXERR && wend <= zlemetacs);
    noaliases = ona;
    strinend();
    inpop();
    errflag &= ~ERRFLAG_ERROR;
    noerrs = ne;
    zcontext_restore();
    zlemetacs = ocs;
    wb = owb;
    we = owe;
    addedx = oadx;

    /* convert offsets for mark and zlecs back to ZLE internal format */
    linein[wend] = '\0'; /* a bit of a hack to get two offsets */
    free(stringaszleline(linein, wstarts[wcur], &zlecs, &tmpsz, &mark));
    free(linein);

    if (IS_THINGY(bindk, selectinshellword)) {
	ZLE_CHAR_T *match = ZWS("`\'\"");
	ZLE_CHAR_T *lmatch = ZWS("\'({"), *rmatch = ZWS("\')}");
	ZLE_CHAR_T *ematch = match, *found;
	int start, end = zlecs;
	/* for 'in' widget, don't include initial blanks ... */
	while (mark < zlecs && ZC_iblank(zleline[mark]))
	    INCPOS(mark);
	/* ... or a matching pair of quotes */
	start = mark;
	if (zleline[start] == ZWC('$')) {
	    match = lmatch;
	    ematch = rmatch;
	    INCPOS(start);
	}
	found = ZS_strchr(match, zleline[start]);
	if (found) {
	    DECPOS(end);
	    if (zleline[end] == ematch[found-match]) {
		zlecs = end;
		INCPOS(start);
		mark = start;
	    }
	}
    }

    /* Adjustment: vi operators don't include the cursor position */
    if (!virangeflag && invicmdmode())
       DECCS();

    return 0;
}
