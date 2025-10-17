/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#define isQUIT(c)     ((c) == QUIT || (c) == ESCAPE)

#define key_RECUR     CTRL('W')
#define key_NEWLINE   CTRL('N')
#define key_BACKSPACE '\b'

static FILE *linedata;

static void
failed(const char *s)
{
    perror(s);
    ExitProgram(EXIT_FAILURE);
}

static void
init_linedata(const char *name)
{
    if ((linedata = fopen(name, "r")) == 0) {
	failed(name);
    }
}

static int
read_linedata(WINDOW *work)
{
    int result;
    if (linedata != 0) {
	result = fgetc(linedata);
	if (result == EOF) {
	    fclose(linedata);
	    linedata = 0;
	    result = read_linedata(work);
	} else {
	    wrefresh(work);
	    if (result == '\n') {
		result = key_NEWLINE;
	    }
	}
    } else {
#ifdef WIDE_LINEDATA
	wint_t ch;
	int code;

	result = ERR;
	while ((code = wget_wch(work, &ch)) != ERR) {

	    if (code == KEY_CODE_YES) {
		switch (ch) {
		case KEY_DOWN:
		    result = key_NEWLINE;
		    break;
		case KEY_BACKSPACE:
		    result = key_BACKSPACE;
		    break;
		default:
		    beep();
		    continue;
		}
	    } else {
		result = (int) ch;
		break;
	    }
	}
#else
	result = wgetch(work);
#endif
    }
    return result;
}
