/*
botw-qt-backend
Copyright (C) 2023 iTNTPiston

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>
#include <opencv2/opencv.hpp>
#include "botwqt-metadata.generated.h"
#include "botwqt-frame.h"
#include "botwqt.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

static bool botwqt_is_format_supported(enum video_format format)
{
	switch (format) {
		case VIDEO_FORMAT_YUY2:
			return true;
		default:
			return false;
	}
}

namespace botwqt {

Backend::Backend() {
	m_frame = new cv::Mat();
	m_frame_valid = false;

}

Backend::~Backend() {
	delete m_frame;
}

void Backend::update_settings(obs_data_t *p_settings) {
	m_settings.enable_preview = (bool) obs_data_get_bool(p_settings, BOTW_QT_SETTING_PREVIEW);
	m_settings.threshold = (uint32_t) obs_data_get_int(p_settings, BOTW_QT_SETTING_THRESHOLD);
	m_settings.interval = (uint32_t) obs_data_get_int(p_settings, BOTW_QT_SETTING_RATE);

	auto port = (uint16_t) obs_data_get_int(p_settings, BOTW_QT_SETTING_PORT);

	if (port != m_settings.port) {
		m_settings.port = port;
		m_server.stop();
		m_server.start(m_settings.port);
	}
}

void Backend::process_frame(struct obs_source_frame *p_frame){
	// For unsupported formats, return immediately
	if (!botwqt_is_format_supported(p_frame->format)) {
		return;
	}
	// If the interval is not reached, return
	if(m_count_since_last < m_settings.interval){
		m_count_since_last++;
		// Still update the preview with the last processed frame if needed
		if (m_settings.enable_preview) {
			cv::Mat frame_wrapper;
			init_frame_wrapper(p_frame, frame_wrapper);
			overlay_preview(p_frame, frame_wrapper);
		}
		return;
	}
	m_count_since_last = 0;
	
	// Process a new frame
	cv::Mat frame_wrapper;
	init_frame_wrapper(p_frame, frame_wrapper);
	process_image(p_frame, frame_wrapper);
	overlay_preview(p_frame, frame_wrapper);
	
	// Send the frame to client
	if(m_frame_valid){
		Frame f(*m_frame);
		m_server.send_frame(f);
	}
}

void Backend::init_frame_wrapper(struct obs_source_frame* p_frame, cv::Mat& frame_wrapper) {
	switch(p_frame->format) {
		case VIDEO_FORMAT_YUY2: {
			frame_wrapper = cv::Mat(p_frame->height, p_frame->width, CV_8UC2, p_frame->data[0]);
			break;
		}
			
		default:
			break;
	}
}

void Backend::process_image(struct obs_source_frame* p_frame, cv::Mat& frame_wrapper) {
	cv::Rect crop = cv::Rect(
		BOTW_QT_BANNER_X(p_frame->width),
		BOTW_QT_BANNER_Y(p_frame->height),
		BOTW_QT_BANNER_W(p_frame->width),
		BOTW_QT_BANNER_H(p_frame->height)
	);

	m_frame_valid = true;

	switch(p_frame->format) {
		case VIDEO_FORMAT_YUY2: {
			frame_wrapper = cv::Mat(p_frame->height, p_frame->width, CV_8UC2, p_frame->data[0]);
			cv::Mat cropped = frame_wrapper(crop);
			cv::cvtColor(cropped, *m_frame, cv::COLOR_YUV2GRAY_YUY2);
			cv::threshold(*m_frame, *m_frame, m_settings.threshold, 255, cv::THRESH_BINARY);
			break;
		}
			
		default:
			m_frame_valid = false;
			break;
	}
}

void Backend::overlay_preview(struct obs_source_frame* p_frame, cv::Mat& frame_wrapper) {
	if (!m_frame_valid) {
		return;
	}
	if (!m_settings.enable_preview) {
		return;
	}
	int crop_start_y = BOTW_QT_BANNER_Y(p_frame->height);
	int crop_start_x = BOTW_QT_BANNER_X(p_frame->width);
	switch(p_frame->format) {
		case VIDEO_FORMAT_YUY2:
			for (int x=0; x<m_frame->cols && x+crop_start_x<frame_wrapper.cols; x++) {
				for (int y=0; y<m_frame->rows  && y+crop_start_y<frame_wrapper.rows; y++) {
					frame_wrapper.at<cv::Vec2b>(y+crop_start_y, x+crop_start_x) = cv::Vec2b(m_frame->at<uint8_t>(y, x), 0x80);
				}
			}
			break;

		default:
			break;
	}
}

}

static const char* botwqt_get_name(void *)
{
	return "BotW Quest Tracker Backend";
}

static void botwqt_update_settings(void *p_backend, obs_data_t *p_settings)
{
	reinterpret_cast<botwqt::Backend*>(p_backend)->update_settings(p_settings);
}

static void* botwqt_create(obs_data_t *p_settings, obs_source_t */*p_context*/)
{
	botwqt::Backend *p_backend = new botwqt::Backend();
	p_backend->update_settings(p_settings);

	return reinterpret_cast<void*>(p_backend);
}

static void botwqt_destroy(void *data)
{
	auto p_backend = reinterpret_cast<botwqt::Backend*>(data);
	delete p_backend;
}

static void botwqt_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, BOTW_QT_SETTING_PREVIEW, true);
	obs_data_set_default_int(settings, BOTW_QT_SETTING_RATE, 10);
	obs_data_set_default_int(settings, BOTW_QT_SETTING_THRESHOLD, 60);
	obs_data_set_default_int(settings, BOTW_QT_SETTING_PORT, 8899);
}

static obs_properties_t* botwqt_get_properties(void *)
{
	obs_properties_t* props = obs_properties_create();

	obs_properties_add_text(props, BOTW_QT_SETTING_MODULE_TEXT(FORMAT_NOTE), OBS_TEXT_INFO);
	obs_properties_add_bool(props, BOTW_QT_SETTING_MODULE_TEXT(PREVIEW));
	obs_property_t *p = obs_properties_add_int_slider(props, BOTW_QT_SETTING_MODULE_TEXT(RATE), 5, 60, 1);
	obs_property_int_set_suffix(p, BOTW_QT_MODULE_TEXT(RATE_UNIT));
	obs_properties_add_int_slider(props, BOTW_QT_SETTING_MODULE_TEXT(THRESHOLD), 0, 255, 1);
	obs_properties_add_int(props, BOTW_QT_SETTING_MODULE_TEXT(PORT), 2000, 65535, 1);

	return props;
}


static struct obs_source_frame* botwqt_filter_video(void *p_backend, struct obs_source_frame *p_frame)
{
	reinterpret_cast<botwqt::Backend*>(p_backend)->process_frame(p_frame);
	return p_frame;
}

struct obs_source_info botwqt_backend {
	.id = "botwqt_backend",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
	.get_name = botwqt_get_name,
	.create = botwqt_create,
	.destroy = botwqt_destroy,
	.get_defaults = botwqt_get_defaults,
	.get_properties = botwqt_get_properties,
	.update = botwqt_update_settings,
	.filter_video = botwqt_filter_video,
};

bool obs_module_load(void)
{
	blog(LOG_INFO, "registering backend as video filter");
	obs_register_source(&botwqt_backend);

	blog(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);
	return true;
}

void obs_module_unload()
{
	blog(LOG_INFO, "plugin unloaded");
}