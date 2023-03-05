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

#ifndef _BOTWQT_H_
#define _BOTWQT_H_

#define BOTW_QT_SETTING_PREVIEW "botwqt_preview"
#define BOTW_QT_SETTING_RATE "botwqt_rate"
#define BOTW_QT_SETTING_RATE_UNIT "botwqt_rate_unit"
#define BOTW_QT_SETTING_THRESHOLD "botwqt_threshold"
#define BOTW_QT_SETTING_PORT "botwqt_port"
#define BOTW_QT_SETTING_FORMAT_NOTE "botwqt_format_note"

#define BOTW_QT_MODULE_TEXT(name) obs_module_text(BOTW_QT_SETTING_##name)
#define BOTW_QT_SETTING_MODULE_TEXT(name) BOTW_QT_SETTING_##name, BOTW_QT_MODULE_TEXT(name)

// Banner Position Percentage
#define BOTW_QT_BANNER_X(width) ((width) * 23 / 100)
#define BOTW_QT_BANNER_Y(height) ((height) * 19 / 100)
#define BOTW_QT_BANNER_W(width) ((width) * 54 / 100)
#define BOTW_QT_BANNER_H(height) ((height) * 9 / 100)
#include "botwqt-settings.h"
#include "botwqt-server.h"

// OBS Forward Declarations
struct obs_data;
struct obs_source_frame;
typedef struct obs_data obs_data_t;

// OpenCV Forward Declarations
namespace cv {
class Mat;
}

namespace botwqt {

class Backend {
public:
    Backend();
    ~Backend();

    void update_settings(obs_data_t *p_settings);
    void process_frame(struct obs_source_frame *p_frame);

private:
    // settings
    Settings m_settings;
    // temporary frame storage
    bool m_frame_valid = false;
    cv::Mat *m_frame = nullptr;
    // Server instance
    Server m_server;
    // How many frames since last detection
    uint32_t m_count_since_last = 0;

    // Initialize a cv::Mat to wrap the frame data
    void init_frame_wrapper(struct obs_source_frame* p_frame, cv::Mat& frame_wrapper);

    // Process the image and cache it
    void process_image(struct obs_source_frame* p_frame, cv::Mat& frame_wrapper);

    // Overlay the latest processed image on the preview
    void overlay_preview(struct obs_source_frame* p_frame, cv::Mat& frame_wrapper);
};
}

#endif // _BOTWQT_H_
