/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
 * decomment.c
 *
 * Removes all comments and (optionally) whitespace from an input file.
 * Writes result on stdout.
 */

#include <stdio.h>
#include <ctype.h>      /* for isspace */
#include <libc.h>

/*
 * State of input scanner.
 */
typedef enum {
	IS_NORMAL,
	IS_SLASH,               // encountered opening '/'
	IS_IN_COMMENT,          // within / * * / comment
	IS_STAR,                // encountered closing '*'
	IS_IN_END_COMMENT       // within / / comment
} input_state_t;

static void usage(char **argv);

int
main(int argc, char **argv)
{
	FILE *fp;
	char bufchar;
	input_state_t input_state = IS_NORMAL;
	int exit_code = 0;
	int remove_whitespace = 0;
	int arg;

	if (argc < 2) {
		usage(argv);
	}
	for (arg = 2; arg < argc; arg++) {
		switch (argv[arg][0]) {
		case 'r':
			remove_whitespace++;
			break;
		default:
			usage(argv);
		}
	}

	fp = fopen(argv[1], "r");
	if (!fp) {
		fprintf(stderr, "Error opening %s\n", argv[1]);
		perror("fopen");
		exit(1);
	}
	for (;;) {
		bufchar = getc_unlocked(fp);
		if (bufchar == EOF) {
			break;
		}

		switch (input_state) {
		case IS_NORMAL:
			if (bufchar == '/') {
				/*
				 * Might be start of a comment.
				 */
				input_state = IS_SLASH;
			} else {
				if (!(remove_whitespace && isspace(bufchar))) {
					putchar_unlocked(bufchar);
				}
			}
			break;

		case IS_SLASH:
			switch (bufchar) {
			case '*':
				/*
				 * Start of normal comment.
				 */
				input_state = IS_IN_COMMENT;
				break;

			case '/':
				/*
				 * Start of 'to-end-of-line' comment.
				 */
				input_state = IS_IN_END_COMMENT;
				break;

			default:
				/*
				 * Not the start of comment. Emit the '/'
				 * we skipped last char in case we were
				 * entering a comment this time, then the
				 * current char.
				 */
				putchar_unlocked('/');
				if (!(remove_whitespace && isspace(bufchar))) {
					putchar_unlocked(bufchar);
				}
				input_state = IS_NORMAL;
				break;
			}
			break;

		case IS_IN_COMMENT:
			if (bufchar == '*') {
				/*
				 * Maybe ending comment...
				 */
				input_state = IS_STAR;
			}
			break;


		case IS_STAR:
			switch (bufchar) {
			case '/':
				/*
				 * End of normal comment.
				 */
				input_state = IS_NORMAL;
				break;

			case '*':
				/*
				 * Still could be one char away from end
				 * of comment.
				 */
				break;

			default:
				/*
				 * Still inside comment, no end in sight.
				 */
				input_state = IS_IN_COMMENT;
				break;
			}
			break;

		case IS_IN_END_COMMENT:
			if (bufchar == '\n') {
				/*
				 * End of comment. Emit the newline if
				 * appropriate.
				 */
				if (!remove_whitespace) {
					putchar_unlocked(bufchar);
				}
				input_state = IS_NORMAL;
			}
			break;
		} /* switch input_state */
	}         /* main read loop */

	/*
	 * Done.
	 */
	return exit_code;
}

static void
usage(char **argv)
{
	printf("usage: %s infile [r(emove whitespace)]\n", argv[0]);
	exit(1);
}
