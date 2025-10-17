/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#ifndef SUDO_FATAL_H
#define	SUDO_FATAL_H

#include <stdarg.h>
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif /* HAVE_STDBOOL_H */

#include "sudo_plugin.h"	/* for conversation function */

/* No output to debug files when fuzzing. */
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
# define SUDO_ERROR_WRAP 0
#endif

/*
 * We wrap fatal/fatalx and warn/warnx so that the same output can
 * go to the debug file, if there is one.
 */
#if (defined(SUDO_ERROR_WRAP) && SUDO_ERROR_WRAP == 0) || defined(NO_VARIADIC_MACROS)
# define sudo_fatal sudo_fatal_nodebug_v1
# define sudo_fatalx sudo_fatalx_nodebug_v1
# define sudo_gai_fatal sudo_gai_fatal_nodebug_v1
# define sudo_warn sudo_warn_nodebug_v1
# define sudo_warnx sudo_warnx_nodebug_v1
# define sudo_gai_warn sudo_gai_warn_nodebug_v1
# define sudo_vfatal(fmt, ap) sudo_vfatal_nodebug_v1((fmt), (ap))
# define sudo_vfatalx(fmt, ap) sudo_vfatalx_nodebug_v1((fmt), (ap))
# define sudo_gai_vfatal(en, fmt, ap) sudo_vfatal_nodebug_v1((en), (fmt), (ap))
# define sudo_vwarn(fmt, ap) sudo_vwarn_nodebug_v1((fmt), (ap))
# define sudo_vwarnx(fmt, ap) sudo_vwarnx_nodebug_v1((fmt), (ap))
# define sudo_gai_vwarn(en, fmt, ap) sudo_vwarn_nodebug_v1((en), (fmt), (ap))
#else /* SUDO_ERROR_WRAP */
# if defined(__GNUC__) && __GNUC__ == 2
#  define sudo_fatal(fmt...) do {					       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|SUDO_DEBUG_ERRNO|sudo_debug_subsys, \
	fmt);								       \
    sudo_fatal_nodebug_v1(fmt);						       \
} while (0)
#  define sudo_fatalx(fmt...) do {					       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|sudo_debug_subsys, fmt);	       \
    sudo_fatalx_nodebug_v1(fmt);					       \
} while (0)
#  define sudo_gai_fatal(en, fmt...) do {				       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|sudo_debug_subsys, fmt);	       \
    sudo_gai_fatal_nodebug_v1((en), fmt);				       \
} while (0)
#  define sudo_warn(fmt...) do {					       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|SUDO_DEBUG_ERRNO|sudo_debug_subsys, \
	fmt);								       \
    sudo_warn_nodebug_v1(fmt);						       \
} while (0)
#  define sudo_warnx(fmt...) do {					       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|sudo_debug_subsys, fmt);	       \
    sudo_warnx_nodebug_v1(fmt);						       \
} while (0)
#  define sudo_gai_warn(en, fmt...) do {				       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|sudo_debug_subsys, fmt);	       \
    sudo_gai_warn_nodebug_v1((en), fmt);				       \
} while (0)
# else
#  define sudo_fatal(...) do {						       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|SUDO_DEBUG_ERRNO|sudo_debug_subsys, \
	__VA_ARGS__);							       \
    sudo_fatal_nodebug_v1(__VA_ARGS__);					       \
} while (0)
#  define sudo_fatalx(...) do {						       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|sudo_debug_subsys, __VA_ARGS__);    \
    sudo_fatalx_nodebug_v1(__VA_ARGS__);				       \
} while (0)
#  define sudo_gai_fatal(en, ...) do {					       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|sudo_debug_subsys, __VA_ARGS__);    \
    sudo_gai_fatal_nodebug_v1((en), __VA_ARGS__);			       \
} while (0)
#  define sudo_warn(...) do {						       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_WARN|SUDO_DEBUG_LINENO|SUDO_DEBUG_ERRNO|sudo_debug_subsys,  \
	__VA_ARGS__);							       \
    sudo_warn_nodebug_v1(__VA_ARGS__);					       \
} while (0)
#  define sudo_warnx(...) do {						       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_WARN|SUDO_DEBUG_LINENO|sudo_debug_subsys, __VA_ARGS__);     \
    sudo_warnx_nodebug_v1(__VA_ARGS__);					       \
} while (0)
#  define sudo_gai_warn(en, ...) do {					       \
    sudo_debug_printf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_WARN|SUDO_DEBUG_LINENO|sudo_debug_subsys, __VA_ARGS__);     \
    sudo_gai_warn_nodebug_v1((en), __VA_ARGS__);			       \
} while (0)
# endif /* __GNUC__ == 2 */
# define sudo_vfatal(fmt, ap) do {					       \
    va_list ap2;							       \
    va_copy(ap2, (ap));							       \
    sudo_debug_vprintf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|SUDO_DEBUG_ERRNO|sudo_debug_subsys, \
	(fmt), ap2);							       \
    sudo_vfatal_nodebug_v1((fmt), (ap));					       \
} while (0)
# define sudo_vfatalx(fmt, ap) do {					       \
    va_list ap2;							       \
    va_copy(ap2, (ap));							       \
    sudo_debug_vprintf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|sudo_debug_subsys, (fmt), ap2);     \
    sudo_vfatalx_nodebug_v1((fmt), (ap));				       \
} while (0)
# define sudo_gai_vfatal(en, fmt, ap) do {				       \
    va_list ap2;							       \
    va_copy(ap2, (ap));							       \
    sudo_debug_vprintf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO|sudo_debug_subsys, (fmt), ap2);     \
    sudo_gai_vfatal_nodebug_v1((en), (fmt), (ap));			       \
} while (0)
# define sudo_vwarn(fmt, ap) do {					       \
    va_list ap2;							       \
    va_copy(ap2, (ap));							       \
    sudo_debug_vprintf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_WARN|SUDO_DEBUG_LINENO|SUDO_DEBUG_ERRNO|sudo_debug_subsys,  \
	(fmt), ap2);							       \
    sudo_vwarn_nodebug_v1((fmt), (ap));					       \
} while (0)
# define sudo_vwarnx(fmt, ap) do {					       \
    va_list ap2;							       \
    va_copy(ap2, (ap));							       \
    sudo_debug_vprintf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_WARN|SUDO_DEBUG_LINENO|sudo_debug_subsys, (fmt), ap2);      \
    sudo_vwarnx_nodebug_v1((fmt), (ap));				       \
} while (0)
# define sudo_gai_vwarn(en, fmt, ap) do {				       \
    va_list ap2;							       \
    va_copy(ap2, (ap));							       \
    sudo_debug_vprintf2(__func__, __FILE__, __LINE__,			       \
	SUDO_DEBUG_WARN|SUDO_DEBUG_LINENO|sudo_debug_subsys, (fmt), ap2);      \
    sudo_gai_vwarn_nodebug_v1((en), (fmt), (ap));			       \
} while (0)
#endif /* SUDO_ERROR_WRAP */

typedef void (*sudo_fatal_callback_t)(void);
typedef bool (*sudo_warn_setlocale_t)(bool, int *);

sudo_dso_public int  sudo_fatal_callback_deregister_v1(sudo_fatal_callback_t func);
sudo_dso_public int  sudo_fatal_callback_register_v1(sudo_fatal_callback_t func);
sudo_dso_public char *sudo_warn_gettext_v1(const char *domainname, const char *msgid) sudo_attr_fmt_arg(2);
sudo_dso_public void sudo_warn_set_locale_func_v1(sudo_warn_setlocale_t func);
sudo_noreturn sudo_dso_public void sudo_fatal_nodebug_v1(const char *fmt, ...) sudo_printf0like(1, 2);
sudo_noreturn sudo_dso_public void sudo_fatalx_nodebug_v1(const char *fmt, ...) sudo_printflike(1, 2);
sudo_noreturn sudo_dso_public void sudo_gai_fatal_nodebug_v1(int errnum, const char *fmt, ...) sudo_printflike(2, 3);
sudo_noreturn sudo_dso_public void sudo_vfatal_nodebug_v1(const char *fmt, va_list ap) sudo_printf0like(1, 0);
sudo_noreturn sudo_dso_public void sudo_vfatalx_nodebug_v1(const char *fmt, va_list ap) sudo_printflike(1, 0);
sudo_noreturn sudo_dso_public void sudo_gai_vfatal_nodebug_v1(int errnum, const char *fmt, va_list ap) sudo_printflike(2, 0);
sudo_dso_public void sudo_warn_nodebug_v1(const char *fmt, ...) sudo_printf0like(1, 2);
sudo_dso_public void sudo_warnx_nodebug_v1(const char *fmt, ...) sudo_printflike(1, 2);
sudo_dso_public void sudo_gai_warn_nodebug_v1(int errnum, const char *fmt, ...) sudo_printflike(2, 3);
sudo_dso_public void sudo_vwarn_nodebug_v1(const char *fmt, va_list ap) sudo_printf0like(1, 0);
sudo_dso_public void sudo_vwarnx_nodebug_v1(const char *fmt, va_list ap) sudo_printflike(1, 0);
sudo_dso_public void sudo_gai_vwarn_nodebug_v1(int errnum, const char *fmt, va_list ap) sudo_printflike(2, 0);
sudo_dso_public void sudo_warn_set_conversation_v1(sudo_conv_t conv);

#define sudo_fatal_callback_deregister(_a) sudo_fatal_callback_deregister_v1((_a))
#define sudo_fatal_callback_register(_a) sudo_fatal_callback_register_v1((_a))
#define sudo_warn_set_locale_func(_a) sudo_warn_set_locale_func_v1((_a))
#define sudo_fatal_nodebug sudo_fatal_nodebug_v1
#define sudo_fatalx_nodebug sudo_fatalx_nodebug_v1
#define sudo_gai_fatal_nodebug sudo_gai_fatal_nodebug_v1
#define sudo_vfatal_nodebug(_a, _b) sudo_vfatal_nodebug_v1((_a), (_b))
#define sudo_vfatalx_nodebug(_a, _b) sudo_vfatalx_nodebug_v1((_a), (_b))
#define sudo_gai_vfatal_nodebug(_a, _b, _c) sudo_gai_vfatal_nodebug_v1((_a), (_b), (_c))
#define sudo_warn_nodebug sudo_warn_nodebug_v1
#define sudo_warnx_nodebug sudo_warnx_nodebug_v1
#define sudo_gai_warn_nodebug sudo_gai_warn_nodebug_v1
#define sudo_vwarn_nodebug(_a, _b) sudo_vwarn_nodebug_v1((_a), (_b))
#define sudo_vwarnx_nodebug(_a, _b) sudo_vwarnx_nodebug_v1((_a), (_b))
#define sudo_gai_vwarn_nodebug(_a, _b, _c) sudo_gai_vwarn_nodebug_v1((_a), (_b), (_c))
#define sudo_warn_set_conversation(_a) sudo_warn_set_conversation_v1(_a)

#ifdef DEFAULT_TEXT_DOMAIN
# define sudo_warn_gettext(_a) sudo_warn_gettext_v1(DEFAULT_TEXT_DOMAIN, (_a))
#else
# define sudo_warn_gettext(_a) sudo_warn_gettext_v1(NULL, (_a))
#endif

#endif /* SUDO_FATAL_H */
