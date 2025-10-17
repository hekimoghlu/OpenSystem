/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
#include "libusbi.h"

#include <ctype.h>
#include <string.h>

/** \ingroup libusb_misc
 * How to add a new \ref libusb_strerror() translation:
 * <ol>
 * <li> Download the latest \c strerror.c from:<br>
 *      https://raw.github.com/libusb/libusb/master/libusb/strerror.c </li>
 * <li> Open the file in an UTF-8 capable editor </li>
 * <li> Add the 2 letter <a href="http://en.wikipedia.org/wiki/List_of_ISO_639-1_codes">ISO 639-1</a>
 *      code for your locale at the end of \c usbi_locale_supported[]<br>
 *    Eg. for Chinese, you would add "zh" so that:
 *    \code... usbi_locale_supported[] = { "en", "nl", "fr" };\endcode
 *    becomes:
 *    \code... usbi_locale_supported[] = { "en", "nl", "fr", "zh" };\endcode </li>
 * <li> Copy the <tt>{ / * English (en) * / ... }</tt> section and add it at the end of \c usbi_localized_errors<br>
 *    Eg. for Chinese, the last section of \c usbi_localized_errors could look like:
 *    \code
 *     }, { / * Chinese (zh) * /
 *         "Success",
 *         ...
 *         "Other error",
 *     },
 * };\endcode </li>
 * <li> Translate each of the English messages from the section you copied into your language </li>
 * <li> Save the file (in UTF-8 format) and send it to \c libusb-devel\@lists.sourceforge.net </li>
 * </ol>
 */

static const char * const usbi_locale_supported[] = { "en", "nl", "fr", "ru", "de", "hu" };
static const char * const usbi_localized_errors[ARRAYSIZE(usbi_locale_supported)][LIBUSB_ERROR_COUNT] = {
	{ /* English (en) */
		"Success",
		"Input/Output Error",
		"Invalid parameter",
		"Access denied (insufficient permissions)",
		"No such device (it may have been disconnected)",
		"Entity not found",
		"Resource busy",
		"Operation timed out",
		"Overflow",
		"Pipe error",
		"System call interrupted (perhaps due to signal)",
		"Insufficient memory",
		"Operation not supported or unimplemented on this platform",
		"Other error",
	}, { /* Dutch (nl) */
		"Gelukt",
		"Invoer-/uitvoerfout",
		"Ongeldig argument",
		"Toegang geweigerd (onvoldoende toegangsrechten)",
		"Apparaat bestaat niet (verbinding met apparaat verbroken?)",
		"Niet gevonden",
		"Apparaat of hulpbron is bezig",
		"Bewerking verlopen",
		"Waarde is te groot",
		"Gebroken pijp",
		"Onderbroken systeemaanroep",
		"Onvoldoende geheugen beschikbaar",
		"Bewerking wordt niet ondersteund",
		"Andere fout",
	}, { /* French (fr) */
		"SuccÃ¨s",
		"Erreur d'entrÃ©e/sortie",
		"ParamÃ¨tre invalide",
		"AccÃ¨s refusÃ© (permissions insuffisantes)",
		"PÃ©riphÃ©rique introuvable (peut-Ãªtre dÃ©connectÃ©)",
		"ElÃ©ment introuvable",
		"Resource dÃ©jÃ  occupÃ©e",
		"Operation expirÃ©e",
		"DÃ©bordement",
		"Erreur de pipe",
		"Appel systÃ¨me abandonnÃ© (peut-Ãªtre Ã  cause dâ€™un signal)",
		"MÃ©moire insuffisante",
		"OpÃ©ration non supportÃ©e or non implÃ©mentÃ©e sur cette plateforme",
		"Autre erreur",
	}, { /* Russian (ru) */
		"Ð£ÑÐ¿ÐµÑ…",
		"ÐžÑˆÐ¸Ð±ÐºÐ° Ð²Ð²Ð¾Ð´Ð°/Ð²Ñ‹Ð²Ð¾Ð´Ð°",
		"ÐÐµÐ²ÐµÑ€Ð½Ñ‹Ð¹ Ð¿Ð°Ñ€Ð°Ð¼ÐµÑ‚Ñ€",
		"Ð”Ð¾ÑÑ‚ÑƒÐ¿ Ð·Ð°Ð¿Ñ€ÐµÑ‰Ñ‘Ð½ (Ð½Ðµ Ñ…Ð²Ð°Ñ‚Ð°ÐµÑ‚ Ð¿Ñ€Ð°Ð²)",
		"Ð£ÑÑ‚Ñ€Ð¾Ð¹ÑÑ‚Ð²Ð¾ Ð¾Ñ‚ÑÑƒÑ‚ÑÑ‚Ð²ÑƒÐµÑ‚ (Ð²Ð¾Ð·Ð¼Ð¾Ð¶Ð½Ð¾, Ð¾Ð½Ð¾ Ð±Ñ‹Ð»Ð¾ Ð¾Ñ‚ÑÐ¾ÐµÐ´Ð¸Ð½ÐµÐ½Ð¾)",
		"Ð­Ð»ÐµÐ¼ÐµÐ½Ñ‚ Ð½Ðµ Ð½Ð°Ð¹Ð´ÐµÐ½",
		"Ð ÐµÑÑƒÑ€Ñ Ð·Ð°Ð½ÑÑ‚",
		"Ð˜ÑÑ‚ÐµÐºÐ»Ð¾ Ð²Ñ€ÐµÐ¼Ñ Ð¾Ð¶Ð¸Ð´Ð°Ð½Ð¸Ñ Ð¾Ð¿ÐµÑ€Ð°Ñ†Ð¸Ð¸",
		"ÐŸÐµÑ€ÐµÐ¿Ð¾Ð»Ð½ÐµÐ½Ð¸Ðµ",
		"ÐžÑˆÐ¸Ð±ÐºÐ° ÐºÐ°Ð½Ð°Ð»Ð°",
		"Ð¡Ð¸ÑÑ‚ÐµÐ¼Ð½Ñ‹Ð¹ Ð²Ñ‹Ð·Ð¾Ð² Ð¿Ñ€ÐµÑ€Ð²Ð°Ð½ (Ð²Ð¾Ð·Ð¼Ð¾Ð¶Ð½Ð¾, ÑÐ¸Ð³Ð½Ð°Ð»Ð¾Ð¼)",
		"ÐŸÐ°Ð¼ÑÑ‚ÑŒ Ð¸ÑÑ‡ÐµÑ€Ð¿Ð°Ð½Ð°",
		"ÐžÐ¿ÐµÑ€Ð°Ñ†Ð¸Ñ Ð½Ðµ Ð¿Ð¾Ð´Ð´ÐµÑ€Ð¶Ð¸Ð²Ð°ÐµÑ‚ÑÑ Ð´Ð°Ð½Ð½Ð¾Ð¹ Ð¿Ð»Ð°Ñ‚Ñ„Ð¾Ñ€Ð¼Ð¾Ð¹",
		"ÐÐµÐ¸Ð·Ð²ÐµÑÑ‚Ð½Ð°Ñ Ð¾ÑˆÐ¸Ð±ÐºÐ°"
	}, { /* German (de) */
		"Erfolgreich",
		"Eingabe-/Ausgabefehler",
		"UngÃ¼ltiger Parameter",
		"Keine Berechtigung (Zugriffsrechte fehlen)",
		"Kein passendes GerÃ¤t gefunden (es kÃ¶nnte entfernt worden sein)",
		"EntitÃ¤t nicht gefunden",
		"Die Ressource ist belegt",
		"Die Wartezeit fÃ¼r die Operation ist abgelaufen",
		"Mehr Daten empfangen als erwartet",
		"DatenÃ¼bergabe unterbrochen (broken pipe)",
		"Unterbrechung wÃ¤hrend des Betriebssystemaufrufs",
		"Nicht genÃ¼gend Hauptspeicher verfÃ¼gbar",
		"Die Operation wird nicht unterstÃ¼tzt oder ist auf dieser Platform nicht implementiert",
		"Allgemeiner Fehler",
	}, { /* Hungarian (hu) */
		"Sikeres",
		"Be-/kimeneti hiba",
		"Ã‰rvÃ©nytelen paramÃ©ter",
		"HozzÃ¡fÃ©rÃ©s megtagadva",
		"Az eszkÃ¶z nem talÃ¡lhatÃ³ (eltÃ¡volÃ­tottÃ¡k?)",
		"Nem talÃ¡lhatÃ³",
		"Az erÅ‘forrÃ¡s foglalt",
		"IdÅ‘tÃºllÃ©pÃ©s",
		"TÃºlcsordulÃ¡s",
		"TÃ¶rÃ¶tt adatcsatorna",
		"RendszerhÃ­vÃ¡s megszakÃ­tva",
		"Nincs elÃ©g memÃ³ria",
		"A mÅ±velet nem tÃ¡mogatott ezen a rendszeren",
		"ÃltalÃ¡nos hiba",
	},
};

static const char * const (*usbi_error_strings)[LIBUSB_ERROR_COUNT] = &usbi_localized_errors[0];

/** \ingroup libusb_misc
 * Set the language, and only the language, not the encoding! used for
 * translatable libusb messages.
 *
 * This takes a locale string in the default setlocale format: lang[-region]
 * or lang[_country_region][.codeset]. Only the lang part of the string is
 * used, and only 2 letter ISO 639-1 codes are accepted for it, such as "de".
 * The optional region, country_region or codeset parts are ignored. This
 * means that functions which return translatable strings will NOT honor the
 * specified encoding.
 * All strings returned are encoded as UTF-8 strings.
 *
 * If libusb_setlocale() is not called, all messages will be in English.
 *
 * The following functions return translatable strings: libusb_strerror().
 * Note that the libusb log messages controlled through libusb_set_debug()
 * are not translated, they are always in English.
 *
 * For POSIX UTF-8 environments if you want libusb to follow the standard
 * locale settings, call libusb_setlocale(setlocale(LC_MESSAGES, NULL)),
 * after your app has done its locale setup.
 *
 * \param locale locale-string in the form of lang[_country_region][.codeset]
 * or lang[-region], where lang is a 2 letter ISO 639-1 code
 * \returns LIBUSB_SUCCESS on success
 * \returns LIBUSB_ERROR_INVALID_PARAM if the locale doesn't meet the requirements
 * \returns LIBUSB_ERROR_NOT_FOUND if the requested language is not supported
 * \returns a LIBUSB_ERROR code on other errors
 */

int API_EXPORTED libusb_setlocale(const char *locale)
{
	size_t i;

	if (!locale || strlen(locale) < 2
	    || (locale[2] != '\0' && locale[2] != '-' && locale[2] != '_' && locale[2] != '.'))
		return LIBUSB_ERROR_INVALID_PARAM;

	for (i = 0; i < ARRAYSIZE(usbi_locale_supported); i++) {
		if (usbi_locale_supported[i][0] == tolower((unsigned char)locale[0])
		    && usbi_locale_supported[i][1] == tolower((unsigned char)locale[1]))
			break;
	}

	if (i == ARRAYSIZE(usbi_locale_supported))
		return LIBUSB_ERROR_NOT_FOUND;

	usbi_error_strings = &usbi_localized_errors[i];

	return LIBUSB_SUCCESS;
}

/** \ingroup libusb_misc
 * Returns a constant string with a short description of the given error code,
 * this description is intended for displaying to the end user and will be in
 * the language set by libusb_setlocale().
 *
 * The returned string is encoded in UTF-8.
 *
 * The messages always start with a capital letter and end without any dot.
 * The caller must not free() the returned string.
 *
 * \param errcode the error code whose description is desired
 * \returns a short description of the error code in UTF-8 encoding
 */
DEFAULT_VISIBILITY const char * LIBUSB_CALL libusb_strerror(int errcode)
{
	int errcode_index = -errcode;

	if (errcode_index < 0 || errcode_index >= LIBUSB_ERROR_COUNT) {
		/* "Other Error", which should always be our last message, is returned */
		errcode_index = LIBUSB_ERROR_COUNT - 1;
	}

	return (*usbi_error_strings)[errcode_index];
}
