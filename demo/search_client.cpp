/***********************************************************************
 * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
 * @file    search_client.cpp
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2022-10-09 15:24
 * @brief
 ***********************************************************************/
#include <glog/logging.h>
#include "puck/puck/puck_index.h"
#include "puck/tinker/tinker_index.h"
#include "tools/string_split.h"
#include "puck/gflags/puck_gflags.h"
#include "puck/base/time.h"

// int read_feature_data(std::string& input_file, std::vector<std::string>& pic_name,
//                       std::vector<std::vector<float> >& doc_feature) {
//     std::ifstream fin;
//     fin.open(input_file.c_str(), std::ios::binary);

//     if (!fin.good()) {
//         LOG(ERROR) << "cann't open output file:" << input_file.c_str();
//         return -1;
//     }

//     int ret = 0;
//     std::string line;

//     pic_name.clear();
//     doc_feature.clear();

//     while (std::getline(fin, line)) {
//         std::vector<std::string> split_str;

//         if (puck::s_split(line, "\t", split_str) < 2) {
//             LOG(ERROR) << "id:" << pic_name.size() << " get name error.";
//             ret = -1;
//             break;
//         }

//         pic_name.push_back(split_str[0]);
//         std::string feature_str = split_str[1];

//         puck::s_split(feature_str, " ", split_str);

//         std::vector<float> cur_feature;

//         for (u_int32_t idx = 0; idx < split_str.size(); ++idx) {
//             cur_feature.push_back(std::atof(split_str[idx].c_str()));
//         }

//         doc_feature.push_back(cur_feature);
//     }

//     fin.close();
//     LOG(INFO) << "total query cnt = " << pic_name.size();
//     return ret;
// }
//TODO: read file in .fvecs format
int read_feature_data(std::string& input_file, std::vector<std::vector<float>>& doc_feature){
    
    std::ifstream in_descriptor(input_file, std::ios::binary);
    if (!in_descriptor.is_open()) {
        return -1;
    }

    int dim = 0;
    in_descriptor.read((char *)&dim, 4);

    in_descriptor.seekg(0, std::ios::end);
    long long file_size = in_descriptor.tellg();
    int count = file_size / (dim + 1) / 4;
    doc_feature.resize(count);

    in_descriptor.seekg(0, std::ios::beg);
    for (int i = 0; i < count; i++) {
        doc_feature[i].resize(dim);
        in_descriptor.seekg(4, std::ios::cur);
        in_descriptor.read((char *)(doc_feature[i].data()), dim * sizeof(float));
    }
    in_descriptor.close();

    // std::cout<<"number of query vectors = " << count <<std::endl;
    return 0;
}

// TODO: read ground truth
int read_gt_file(const std::string &path, std::vector<int> &groundtruth) {
    std::ifstream in_descriptor(path, std::ios::binary);
    if (!in_descriptor.is_open()) {
        exit(1);
    }
    int k_of_groundtruth = 0;
    in_descriptor.read((char *)&k_of_groundtruth, 4);

    in_descriptor.seekg(0, std::ios::end);
    long long file_size = in_descriptor.tellg();
    int num_of_nodes = file_size / (k_of_groundtruth + 1) / sizeof(int);
    groundtruth.resize(num_of_nodes * k_of_groundtruth);

    in_descriptor.seekg(0, std::ios::beg);
    for (int i = 0; i < num_of_nodes; i++) {
        in_descriptor.seekg(4, std::ios::cur);
        in_descriptor.read((char *)(groundtruth.data() + i * k_of_groundtruth),
                           k_of_groundtruth * sizeof(int));
    }

    in_descriptor.close();

    return k_of_groundtruth;
}

//TODO: compute recall
float compute_recall(int *results, int *groundtruth,
                        int num_of_queries, int computed_k, int gt_k,
                        int actual_k) {
    int num_of_right_candidates = 0;
    for (int i = 0; i < num_of_queries; i++) {
        for (int j = 0; j < computed_k; j++) {
            int res_id = results[i * actual_k + j];

            int *pos_of_candidate = NULL;
            int gt_offset = i * gt_k;
            pos_of_candidate =
                std::find(groundtruth + gt_offset,
                          groundtruth + gt_offset + computed_k, res_id);
            //* result appears in topK of ground truth
            if (pos_of_candidate != groundtruth + gt_offset + computed_k) {
                num_of_right_candidates++;
            }
        }
    }

    float recall =
        (float)num_of_right_candidates / (num_of_queries * computed_k);
    return recall;
}

void DisplayResultAndGroundTruth(int *result, int num_of_queries, int k,
                                 int *gt, int k_of_gt) {
    for (int i = 0; i < num_of_queries; i += (num_of_queries / 10)) {
        int *temp = result + i * k;
        std::cout << "result [" << i << "]: ";
        for (int j = 0; j < k; j++) {
            std::cout << temp[j] << " ";
        }
        std::cout << std::endl;

        temp = gt + i * k_of_gt;
        std::cout << "ground truth [" << i << "]: ";
        for (int j = 0; j < k; j++) {
            std::cout << temp[j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::cout<<"Debug: gflag\n";

    //1. load index
    std::unique_ptr<puck::Index> index;
    puck::IndexType index_type = puck::load_index_type(); //* load index

    if (index_type == puck::IndexType::TINKER) { //Tinker
        LOG(INFO) << "init index of Tinker";
        index.reset(new puck::TinkerIndex());
    } else if (index_type == puck::IndexType::PUCK) {
        LOG(INFO) << "init index of Puck";
        index.reset(new puck::PuckIndex());
    } else if (index_type == puck::IndexType::HIERARCHICAL_CLUSTER) {
        LOG(INFO) << "init index of Flat";
        index.reset(new puck::HierarchicalClusterIndex());
    } else {
        LOG(INFO) << "init index of Error, Nan type";
        return -1;
    }

    if (index == nullptr) {
        LOG(ERROR) << "create new SearchInterface error.";
        return -2;
    }

    int ret = index->init();

    if (ret != 0) {
        LOG(ERROR) << "SearchInterface init error " << ret;
        return -3;
    }
    //2. read input
    std::string query_file(argv[1]); //* 
    std::string gt_file(argv[2]); //*
    std::vector<std::vector<float> > query_data;

    ret = read_feature_data(query_file, query_data);

    if (ret != 0) {
        LOG(ERROR) << "read_feature_data error:" << ret;
        return -4;
    } else {
        LOG(INFO) << "read_feature_data item:" << query_data.size();
    }

    //3. search
    const int item_count = query_data.size();

    puck::Request request;
    puck::Response response;
    request.topk = 100;

    std::vector<int> query_result(item_count*request.topk);
    response.distance = new float[request.topk];
    response.local_idx = new uint32_t[request.topk];

    double avg_search_time = 0.0;

    for (int i = 0; i < item_count; ++i) {
        request.feature = query_data[i].data();

        puck::base::Timer tm_search;
        tm_search.start();

        ret = index->search(&request, &response);

        tm_search.stop();
        avg_search_time += tm_search.m_elapsed(1.0); // ms

        if (ret != 0) {
            LOG(ERROR) << "search item " << i << " error" << ret;
            break;
        }
        for(int j = 0; j < request.topk; j++){
            query_result[i*request.topk+j] = response.local_idx[j];
        }
        // std::cout<<"min dis = "<<response.distance[0]<<std::endl;
    }
    avg_search_time /= item_count;

    std::vector<int> groundtruth;
    int gt_k = read_gt_file(gt_file, groundtruth);

    float recall=compute_recall(query_result.data(), groundtruth.data(), item_count, request.topk, gt_k, request.topk);

    std::cout << "Recall@"<< request.topk <<" = "<< recall << ", avg. search time = " << avg_search_time << "ms, QPS = " << 1000.0/avg_search_time << "\n";

    // DisplayResultAndGroundTruth(query_result.data(), item_count, request.topk, groundtruth.data(), gt_k);

    delete [] response.distance;
    delete [] response.local_idx;

    return 0;
}
