/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
#include <sys/stat.h>
#include <stdlib.h>

/* Utility library. */

#include <msg.h>
#include <msg_vstream.h>
#include <dict.h>
#include <htable.h>
#include <vstring.h>
#include <vstream.h>
#include <stringops.h>
#include <name_mask.h>
#include <warn_stat.h>
#include <mymalloc.h>

/* Global library. */

#include <mail_params.h>
#include <mail_conf.h>
#include <mail_version.h>
#include <mail_run.h>
#include <mail_dict.h>

/* Application-specific. */

#include <postconf.h>

 /*
  * Global storage. See postconf.h for description.
  */
PCF_PARAM_TABLE *pcf_param_table;
PCF_MASTER_ENT *pcf_master_table;
int     pcf_cmd_mode = PCF_DEF_MODE;

 /*
  * Application fingerprinting.
  */
MAIL_VERSION_STAMP_DECLARE;

 /*
  * This program has so many command-line options that we have to implement a
  * compatibility matrix to weed out the conflicting option combinations, and
  * to alert the user about option combinations that have no effect.
  */

 /*
  * Options that are mutually-exclusive. First entry must specify the major
  * modes. Other entries specify conflicts between option modifiers.
  */
static const int pcf_incompat_options[] = {
    /* Major modes. */
    PCF_SHOW_SASL_SERV | PCF_SHOW_SASL_CLNT | PCF_EXP_DSN_TEMPL \
    |PCF_SHOW_LOCKS | PCF_SHOW_MAPS | PCF_DUMP_DSN_TEMPL | PCF_MAIN_PARAM \
    |PCF_MASTER_ENTRY | PCF_MASTER_FLD | PCF_MASTER_PARAM | PCF_SHOW_TLS,
    /* Modifiers. */
    PCF_SHOW_DEFS | PCF_EDIT_CONF | PCF_SHOW_NONDEF | PCF_COMMENT_OUT \
    |PCF_EDIT_EXCL,
    PCF_FOLD_LINE | PCF_EDIT_CONF | PCF_COMMENT_OUT | PCF_EDIT_EXCL,
    PCF_SHOW_EVAL | PCF_EDIT_CONF | PCF_COMMENT_OUT | PCF_EDIT_EXCL,
    PCF_MAIN_OVER | PCF_SHOW_DEFS | PCF_EDIT_CONF | PCF_COMMENT_OUT \
    |PCF_EDIT_EXCL,
    PCF_HIDE_NAME | PCF_EDIT_CONF | PCF_COMMENT_OUT | PCF_EDIT_EXCL \
    |PCF_HIDE_VALUE,
    0,
};

 /*
  * Options, and the only options that they are compatible with. There must
  * be one entry for each major mode. Other entries specify compatibility
  * between option modifiers.
  */
static const int pcf_compat_options[][2] = {
    /* Major modes. */
    {PCF_SHOW_SASL_SERV, 0},
    {PCF_SHOW_SASL_CLNT, 0},
    {PCF_EXP_DSN_TEMPL, 0},
    {PCF_SHOW_LOCKS, 0},
    {PCF_SHOW_MAPS, 0,},
    {PCF_SHOW_TLS, 0,},
    {PCF_DUMP_DSN_TEMPL, 0},
    {PCF_MAIN_PARAM, (PCF_EDIT_CONF | PCF_EDIT_EXCL | PCF_COMMENT_OUT \
		      |PCF_FOLD_LINE | PCF_HIDE_NAME | PCF_PARAM_CLASS \
		      |PCF_SHOW_EVAL | PCF_SHOW_DEFS | PCF_SHOW_NONDEF \
		      |PCF_MAIN_OVER | PCF_HIDE_VALUE)},
    {PCF_MASTER_ENTRY, (PCF_EDIT_CONF | PCF_EDIT_EXCL | PCF_COMMENT_OUT \
			|PCF_FOLD_LINE | PCF_MAIN_OVER | PCF_SHOW_EVAL)},
    {PCF_MASTER_FLD, (PCF_EDIT_CONF | PCF_FOLD_LINE | PCF_HIDE_NAME \
		      |PCF_MAIN_OVER | PCF_SHOW_EVAL | PCF_HIDE_VALUE)},
    {PCF_MASTER_PARAM, (PCF_EDIT_CONF | PCF_EDIT_EXCL | PCF_FOLD_LINE \
			|PCF_HIDE_NAME | PCF_MAIN_OVER | PCF_SHOW_EVAL \
			|PCF_HIDE_VALUE)},
    /* Modifiers. */
    {PCF_PARAM_CLASS, (PCF_MAIN_PARAM | PCF_SHOW_DEFS | PCF_SHOW_NONDEF)},
    0,
};

 /*
  * Compatibility to string conversion support.
  */
static const NAME_MASK pcf_compat_names[] = {
    "-a", PCF_SHOW_SASL_SERV,
    "-A", PCF_SHOW_SASL_CLNT,
    "-b", PCF_EXP_DSN_TEMPL,
    "-C", PCF_PARAM_CLASS,
    "-d", PCF_SHOW_DEFS,
    "-e", PCF_EDIT_CONF,
    "-f", PCF_FOLD_LINE,
    "-F", PCF_MASTER_FLD,
    "-h", PCF_HIDE_NAME,
    "-H", PCF_HIDE_VALUE,
    "-l", PCF_SHOW_LOCKS,
    "-m", PCF_SHOW_MAPS,
    "-M", PCF_MASTER_ENTRY,
    "-n", PCF_SHOW_NONDEF,
    "-o", PCF_MAIN_OVER,
    "-p", PCF_MAIN_PARAM,
    "-P", PCF_MASTER_PARAM,
    "-t", PCF_DUMP_DSN_TEMPL,
    "-T", PCF_SHOW_TLS,
    "-x", PCF_SHOW_EVAL,
    "-X", PCF_EDIT_EXCL,
    "-#", PCF_COMMENT_OUT,
    0,
};

/* usage - enumerate parameters without compatibility info */

static void usage(const char *progname)
{
    msg_fatal("usage: %s"
	      " [-a (server SASL types)]"
	      " [-A (client SASL types)]"
	      " [-b (bounce templates)]"
	      " [-c config_dir]"
	      " [-c param_class]"
	      " [-d (parameter defaults)]"
	      " [-e (edit configuration)]"
	      " [-f (fold lines)]"
	      " [-F (master.cf fields)]"
	      " [-h (no names)]"
	      " [-H (no values)]"
	      " [-l (lock types)]"
	      " [-m (map types)]"
	      " [-M (master.cf)]"
	      " [-n (non-default parameters)]"
	      " [-o name=value (override parameter value)]"
	      " [-p (main.cf, default)]"
	      " [-P (master.cf parameters)]"
	      " [-t (bounce templates)]"
	      " [-T compile-version|run-version|public-key-algorithms]"
	      " [-v (verbose)]"
	      " [-x (expand parameter values)]"
	      " [-X (exclude)]"
	      " [-# (comment-out)]"
	      " [name...]", progname);
}

/* pcf_check_exclusive_options - complain about mutually-exclusive options */

static void pcf_check_exclusive_options(int optval)
{
    const char *myname = "pcf_check_exclusive_options";
    const int *op;
    int     oval;
    unsigned mask;

    for (op = pcf_incompat_options; (oval = *op) != 0; op++) {
	oval &= optval;
	for (mask = ~0U; (mask & oval) != 0; mask >>= 1) {
	    if ((mask & oval) != oval)
		msg_fatal("specify one of %s",
			  str_name_mask(myname, pcf_compat_names, oval));
	}
    }
}

/* pcf_check_compat_options - complain about incompatible options */

static void pcf_check_compat_options(int optval)
{
    const char *myname = "pcf_check_compat_options";
    VSTRING *buf1 = vstring_alloc(10);
    VSTRING *buf2 = vstring_alloc(10);
    const int (*op)[2];
    int     excess;

    for (op = pcf_compat_options; op[0][0] != 0; op++) {
	if ((optval & *op[0]) != 0
	    && (excess = (optval & ~((*op)[0] | (*op)[1]))) != 0)
	    msg_fatal("with option %s, do not specify %s",
		      str_name_mask_opt(buf1, myname, pcf_compat_names,
					(*op)[0], NAME_MASK_NUMBER),
		      str_name_mask_opt(buf2, myname, pcf_compat_names,
					excess, NAME_MASK_NUMBER));
    }
    vstring_free(buf1);
    vstring_free(buf2);
}

/* main */

int     main(int argc, char **argv)
{
    int     ch;
    int     fd;
    struct stat st;
    ARGV   *ext_argv = 0;
    int     param_class = PCF_PARAM_MASK_CLASS;
    static const NAME_MASK param_class_table[] = {
	"builtin", PCF_PARAM_FLAG_BUILTIN,
	"service", PCF_PARAM_FLAG_SERVICE,
	"user", PCF_PARAM_FLAG_USER,
	"all", PCF_PARAM_MASK_CLASS,
	0,
    };
    ARGV   *override_params = 0;
    const char *pcf_tls_arg = 0;

    /*
     * Fingerprint executables and core dumps.
     */
    MAIL_VERSION_STAMP_ALLOCATE;

    /*
     * Be consistent with file permissions.
     */
    umask(022);

    /*
     * To minimize confusion, make sure that the standard file descriptors
     * are open before opening anything else. XXX Work around for 44BSD where
     * fstat can return EBADF on an open file descriptor.
     */
    for (fd = 0; fd < 3; fd++)
	if (fstat(fd, &st) == -1
	    && (close(fd), open("/dev/null", O_RDWR, 0)) != fd)
	    msg_fatal("open /dev/null: %m");

    /*
     * Set up logging.
     */
    msg_vstream_init(argv[0], VSTREAM_ERR);

    /*
     * Parse JCL.
     */
    while ((ch = GETOPT(argc, argv, "aAbc:C:deEfFhHlmMno:pPtT:vxX#")) > 0) {
	switch (ch) {
	case 'a':
	    pcf_cmd_mode |= PCF_SHOW_SASL_SERV;
	    break;
	case 'A':
	    pcf_cmd_mode |= PCF_SHOW_SASL_CLNT;
	    break;
	case 'b':
	    pcf_cmd_mode |= PCF_EXP_DSN_TEMPL;
	    if (ext_argv)
		msg_fatal("specify one of -b and -t");
	    ext_argv = argv_alloc(2);
	    argv_add(ext_argv, "bounce", "-SVnexpand_templates", (char *) 0);
	    break;
	case 'c':
	    if (setenv(CONF_ENV_PATH, optarg, 1) < 0)
		msg_fatal("out of memory");
	    break;
	case 'C':
	    param_class = name_mask_opt("-C option", param_class_table,
			      optarg, NAME_MASK_ANY_CASE | NAME_MASK_FATAL);
	    break;
	case 'd':
	    pcf_cmd_mode |= PCF_SHOW_DEFS;
	    break;
	case 'e':
	    pcf_cmd_mode |= PCF_EDIT_CONF;
	    break;
	case 'f':
	    pcf_cmd_mode |= PCF_FOLD_LINE;
	    break;
	case 'F':
	    pcf_cmd_mode |= PCF_MASTER_FLD;
	    break;
	case '#':
	    pcf_cmd_mode |= PCF_COMMENT_OUT;
	    break;
	case 'h':
	    pcf_cmd_mode |= PCF_HIDE_NAME;
	    break;
	case 'H':
	    pcf_cmd_mode |= PCF_HIDE_VALUE;
	    break;
	case 'l':
	    pcf_cmd_mode |= PCF_SHOW_LOCKS;
	    break;
	case 'm':
	    pcf_cmd_mode |= PCF_SHOW_MAPS;
	    break;
	case 'M':
	    pcf_cmd_mode |= PCF_MASTER_ENTRY;
	    break;
	case 'n':
	    pcf_cmd_mode |= PCF_SHOW_NONDEF;
	    break;
	case 'o':
	    pcf_cmd_mode |= PCF_MAIN_OVER;
	    if (override_params == 0)
		override_params = argv_alloc(2);
	    argv_add(override_params, optarg, (char *) 0);
	    break;
	case 'p':
	    pcf_cmd_mode |= PCF_MAIN_PARAM;
	    break;
	case 'P':
	    pcf_cmd_mode |= PCF_MASTER_PARAM;
	    break;
	case 't':
	    pcf_cmd_mode |= PCF_DUMP_DSN_TEMPL;
	    if (ext_argv)
		msg_fatal("specify one of -b and -t");
	    ext_argv = argv_alloc(2);
	    argv_add(ext_argv, "bounce", "-SVndump_templates", (char *) 0);
	    break;
	case 'T':
	    if (pcf_cmd_mode & PCF_SHOW_TLS)
		msg_fatal("At most one -T <mode> option may be specified");
	    pcf_cmd_mode |= PCF_SHOW_TLS;
	    pcf_tls_arg = optarg;
	    break;
	case 'x':
	    pcf_cmd_mode |= PCF_SHOW_EVAL;
	    break;
	case 'X':
	    /* This is irreversible, therefore require two-finger action. */
	    pcf_cmd_mode |= PCF_EDIT_EXCL;
	    break;
	case 'v':
	    msg_verbose++;
	    break;
	default:
	    usage(argv[0]);
	}
    }

    /*
     * We don't enforce import_environment consistency in this program.
     * 
     * We don't extract import_environment from main.cf, because the postconf
     * command must be able to extract parameter settings from main.cf before
     * all installation parameters such as mail_owner or setgid_group have a
     * legitimate value.
     * 
     * We would need the functionality of mail_params_init() including all the
     * side effects of populating the CONFIG_DICT with default values so that
     * $name expansion works correctly, but excluding all the parameter value
     * sanity checks so that it would not abort at installation time.
     */

    /*
     * Make all options explicit, before checking their compatibility.
     */
#define PCF_MAIN_OR_MASTER \
	(PCF_MAIN_PARAM | PCF_MASTER_ENTRY | PCF_MASTER_FLD | PCF_MASTER_PARAM)

    if ((pcf_cmd_mode & pcf_incompat_options[0]) == 0)
	pcf_cmd_mode |= PCF_MAIN_PARAM;
    if ((pcf_cmd_mode & PCF_MAIN_OR_MASTER)
	&& argv[optind] && strchr(argv[optind], '='))
	pcf_cmd_mode |= PCF_EDIT_CONF;

    /*
     * Sanity check.
     */
    pcf_check_exclusive_options(pcf_cmd_mode);
    pcf_check_compat_options(pcf_cmd_mode);

    if ((pcf_cmd_mode & PCF_EDIT_CONF) && argc == optind)
	msg_fatal("-e requires name=value argument");

    /*
     * Display bounce template information and exit.
     */
    if (ext_argv) {
	if (argv[optind]) {
	    if (argv[optind + 1])
		msg_fatal("options -b and -t require at most one template file");
	    argv_add(ext_argv, "-o",
		     concatenate(VAR_BOUNCE_TMPL, "=",
				 argv[optind], (char *) 0),
		     (char *) 0);
	}
	/* Grr... */
	argv_add(ext_argv, "-o",
		 concatenate(VAR_QUEUE_DIR, "=", ".", (char *) 0),
		 (char *) 0);
	mail_conf_read();
	mail_run_replace(var_daemon_dir, ext_argv->argv);
	/* NOTREACHED */
    }

    /*
     * If showing map types, show them and exit
     */
    if (pcf_cmd_mode & PCF_SHOW_MAPS) {
	mail_conf_read();
	mail_dict_init();
	pcf_show_maps();
    }

    /*
     * If showing locking methods, show them and exit
     */
    else if (pcf_cmd_mode & PCF_SHOW_LOCKS) {
	pcf_show_locks();
    }

    /*
     * If showing master.cf entries, show them and exit
     */
    else if ((pcf_cmd_mode & (PCF_MASTER_ENTRY | PCF_MASTER_FLD | PCF_MASTER_PARAM))
    && !(pcf_cmd_mode & (PCF_EDIT_CONF | PCF_EDIT_EXCL | PCF_COMMENT_OUT))) {
	pcf_read_master(PCF_FAIL_ON_OPEN_ERROR);
	pcf_read_parameters();
	if (override_params)
	    pcf_set_parameters(override_params->argv);
	pcf_register_builtin_parameters(basename(argv[0]), getpid());
	pcf_register_service_parameters();
	pcf_register_user_parameters();
	if (pcf_cmd_mode & PCF_MASTER_FLD)
	    pcf_show_master_fields(VSTREAM_OUT, pcf_cmd_mode, argc - optind,
				   argv + optind);
	else if (pcf_cmd_mode & PCF_MASTER_PARAM)
	    pcf_show_master_params(VSTREAM_OUT, pcf_cmd_mode, argc - optind,
				   argv + optind);
	else
	    pcf_show_master_entries(VSTREAM_OUT, pcf_cmd_mode, argc - optind,
				    argv + optind);
    }

    /*
     * If showing SASL plug-in types, show them and exit
     */
    else if (pcf_cmd_mode & PCF_SHOW_SASL_SERV) {
	pcf_show_sasl(PCF_SHOW_SASL_SERV);
    } else if (pcf_cmd_mode & PCF_SHOW_SASL_CLNT) {
	pcf_show_sasl(PCF_SHOW_SASL_CLNT);
    }

    /*
     * Show TLS info and exit.
     */
    else if (pcf_cmd_mode & PCF_SHOW_TLS) {
	pcf_show_tls(pcf_tls_arg);
    }

    /*
     * Edit main.cf or master.cf.
     */
    else if (pcf_cmd_mode & (PCF_EDIT_CONF | PCF_COMMENT_OUT | PCF_EDIT_EXCL)) {
	if (optind == argc)
	    msg_fatal("missing service argument");
	if (pcf_cmd_mode & (PCF_MASTER_ENTRY | PCF_MASTER_FLD | PCF_MASTER_PARAM)) {
	    pcf_edit_master(pcf_cmd_mode, argc - optind, argv + optind);
	} else {
	    pcf_edit_main(pcf_cmd_mode, argc - optind, argv + optind);
	}
    }

    /*
     * If showing non-default values, read main.cf.
     */
    else {
	if ((pcf_cmd_mode & PCF_SHOW_DEFS) == 0) {
	    pcf_read_parameters();
	    if (override_params)
		pcf_set_parameters(override_params->argv);
	}
	pcf_register_builtin_parameters(basename(argv[0]), getpid());

	/*
	 * Add service-dependent parameters (service names from master.cf)
	 * and user-defined parameters ($name macros in parameter values in
	 * main.cf and master.cf, but only if those names have a name=value
	 * in main.cf or master.cf).
	 */
	pcf_read_master(PCF_WARN_ON_OPEN_ERROR);
	pcf_register_service_parameters();
	if ((pcf_cmd_mode & PCF_SHOW_DEFS) == 0)
	    pcf_register_user_parameters();

	/*
	 * Show the requested values.
	 */
	pcf_show_parameters(VSTREAM_OUT, pcf_cmd_mode, param_class,
			    argv + optind);

	/*
	 * Flag unused parameters. This makes no sense with "postconf -d",
	 * because that ignores all the user-specified parameters and
	 * user-specified macro expansions in main.cf.
	 */
	if ((pcf_cmd_mode & PCF_SHOW_DEFS) == 0) {
	    pcf_flag_unused_main_parameters();
	    pcf_flag_unused_master_parameters();
	}
    }
    vstream_fflush(VSTREAM_OUT);
    exit(0);
}
