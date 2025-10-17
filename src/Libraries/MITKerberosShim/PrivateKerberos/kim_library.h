/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#ifndef KIM_LIBRARY_H
#define KIM_LIBRARY_H

#include <Kerberos/kim.h>

/*!
 * \defgroup kim_library_reference KIM Library Documentation
 * @{
 */

/*! Do not present user interface */
#define KIM_UI_ENVIRONMENT_NONE 0
/*! Automatically determine what user interface is appropriate (default). */
#define KIM_UI_ENVIRONMENT_AUTO 1
/*! Present a graphical user interface */
#define KIM_UI_ENVIRONMENT_GUI  2
/*! Present a command line user interface */
#define KIM_UI_ENVIRONMENT_CLI  3

/*! An integer describing the type of user interface to use. */
typedef int kim_ui_environment;

/*!
 * \param in_ui_environment   an integer value describing the type of user interface to use.
 * \return On success, #KIM_NO_ERROR.  On failure, an error code representing the failure.
 * \note Set to KIM_UI_ENVIRONMENT_AUTO by default.
 * \brief Tell KIM how to present UI from your application.
 */
kim_error kim_library_set_ui_environment (kim_ui_environment in_ui_environment);

/*!
 * \param in_allow_access   a boolean containing whether or not to touch the user's home directory.
 * \return On success, #KIM_NO_ERROR.  On failure, an error code representing the failure.
 * \note This API is usually used for Kerberos authenticated home directories to prevent a deadlock.
 * \brief Tells KIM whether or not it is allowed to touch the user's home directory.
 */
kim_error kim_library_set_allow_home_directory_access (kim_boolean in_allow_access);

/*!
 * \param in_allow_automatic_prompting   a boolean containing whether or not to prompt automatically.
 * \return On success, #KIM_NO_ERROR.  On failure, an error code representing the failure.
 * \brief Tells KIM whether or not it is allowed to automatically present user interface.
 */
kim_error kim_library_set_allow_automatic_prompting (kim_boolean in_allow_automatic_prompting);

/*!
 * \param in_application_name   a string containing the localized name of your application.
 * \return On success, #KIM_NO_ERROR.  On failure, an error code representing the failure.
 * \note On many operating systems KIM can determine the caller's application
 * name automatically.  This call exists for applications to use when those
 * mechanisms fail or do not exist.
 * \brief Set the name of your application for KIM to use for user interface.
 */
kim_error kim_library_set_application_name (kim_string in_application_name);

/*!@}*/

#endif /* KIM_LIBRARY_H */
