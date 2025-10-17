/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#include <getarg.h>
#include "lex.h"

RCSID("$Id$");

extern FILE *yyin;

static getarg_strings preserve;
static getarg_strings seq;
static getarg_strings extra_data;

int
preserve_type(const char *p)
{
    int i;
    for (i = 0; i < preserve.num_strings; i++)
	if (strcmp(preserve.strings[i], p) == 0)
	    return 1;
    return 0;
}

int
seq_type(const char *p)
{
    int i;
    for (i = 0; i < seq.num_strings; i++)
	if (strcmp(seq.strings[i], p) == 0)
	    return 1;
    return 0;
}

int
extra_data_type(const char *p)
{
    int i;
    for (i = 0; i < extra_data.num_strings; i++)
	if (strcmp(extra_data.strings[i], p) == 0)
	    return 1;
    return 0;
}

const char *fuzzer_string = "";
int fuzzer_flag = 0;
int support_ber = 0;
int template_flag = 0;
int rfc1510_bitstring = 0;
int one_code_file = 0;
int foundation_flag = 0;
char *option_file = NULL;
int parse_units_flag = 1;
char *type_file_string = "krb5-types.h";
int version_flag = 0;
int help_flag = 0;
struct getargs args[] = {
    { "fuzzer", 0, arg_flag, &fuzzer_flag },
    { "template", 0, arg_flag, &template_flag },
    { "encode-rfc1510-bit-string", 0, arg_flag, &rfc1510_bitstring },
    { "decode-dce-ber", 0, arg_flag, &support_ber },
    { "support-ber", 0, arg_flag, &support_ber },
    { "preserve-binary", 0, arg_strings, &preserve },
    { "sequence", 0, arg_strings, &seq },
    { "allow-extra-data", 0, arg_strings, &extra_data},
    { "one-code-file", 0, arg_flag, &one_code_file },
    { "option-file", 0, arg_string, &option_file },
    { "parse-units", 0, arg_negative_flag, &parse_units_flag },
    { "foundation", 0, arg_flag, &foundation_flag },
    { "type-file", 0, arg_string, &type_file_string },
    { "version", 0, arg_flag, &version_flag },
    { "help", 0, arg_flag, &help_flag }
};
int num_args = sizeof(args) / sizeof(args[0]);

static void
usage(int code)
{
    arg_printusage(args, num_args, NULL, "[asn1-file [name]]");
    exit(code);
}

int error_flag;

int
main(int argc, char **argv)
{
    int ret;
    const char *file;
    const char *name = NULL;
    int optidx = 0;
    char **arg = NULL;
    int len = 0, i;

    setprogname(argv[0]);
    if(getarg(args, num_args, argc, argv, &optidx))
	usage(1);
    if(help_flag)
	usage(0);
    if(version_flag) {
	print_version(NULL);
	exit(0);
    }
    if (argc == optidx) {
	file = "stdin";
	name = "stdin";
	yyin = stdin;
    } else {
	file = argv[optidx];
	yyin = fopen (file, "r");
	if (yyin == NULL)
	    err (1, "open %s", file);
	if (argc == optidx + 1) {
	    char *p;
	    name = estrdup(file);
	    p = strrchr(name, '.');
	    if (p)
		*p = '\0';
	} else
	    name = argv[optidx + 1];
    }

    if (foundation_flag && !template_flag) {
	printf("--foundation require --template\n");
	exit(1);
    }

    /*
     * Parse extra options file
     */
    if (option_file) {
	char buf[1024];
	FILE *opt;

	opt = fopen(option_file, "r");
	if (opt == NULL) {
	    perror("open");
	    exit(1);
	}

	arg = calloc(2, sizeof(arg[0]));
	if (arg == NULL) {
	    perror("calloc");
	    exit(1);
	}
	arg[0] = option_file;
	arg[1] = NULL;
	len = 1;

	while (fgets(buf, sizeof(buf), opt) != NULL) {
	    buf[strcspn(buf, "\n\r")] = '\0';

	    arg = realloc(arg, (len + 2) * sizeof(arg[0]));
	    if (arg == NULL) {
		perror("malloc");
		exit(1);
	    }
	    arg[len] = strdup(buf);
	    if (arg[len] == NULL) {
		perror("strdup");
		exit(1);
	    }
	    arg[len + 1] = NULL;
	    len++;
	}
	fclose(opt);

	optidx = 0;
	if(getarg(args, num_args, len, arg, &optidx))
	    usage(1);

	if (len != optidx) {
	    fprintf(stderr, "extra args");
	    exit(1);
	}
    }

    if (fuzzer_flag) {
	if (!template_flag) {
	    printf("can't do fuzzer w/o --template");
	    exit(1);
	}
#ifdef ASN1_FUZZER
	fuzzer_string = "_fuzzer";
#endif
    }


    init_generate (file, name);

    if (one_code_file)
	generate_header_of_codefile(name);

    initsym ();
    ret = yyparse ();
    if(ret != 0 || error_flag != 0)
	exit(1);
    close_generate ();
    if (argc != optidx)
	fclose(yyin);

    if (one_code_file)
	close_codefile();

    if (arg) {
	for (i = 1; i < len; i++)
	    free(arg[i]);
	free(arg);
    }

    return 0;
}
