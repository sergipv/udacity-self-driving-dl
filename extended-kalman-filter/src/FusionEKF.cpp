#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // Measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ <<
      0.0225, 0,
      0, 0.0225;

  // Measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ <<
      0.09, 0, 0,
      0, 0.0009, 0,
      0, 0, 0.09;

  // Measurement matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 
      1, 0, 0, 0,
      0, 1, 0, 0;

  Hj_ = MatrixXd(3, 4);

  // State covariance matrix
  MatrixXd P = MatrixXd(4,4);
  P <<
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1000, 0,
      0, 0, 0, 1000;

  // Transition matrix
  MatrixXd F = MatrixXd(4,4);
  F <<
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1;

  // Vector state
  VectorXd x = VectorXd(4);

  // Process Covariance Matrix
  MatrixXd Q = MatrixXd(4,4);
      
  ekf_.Init(x,
            P,
            F,
            H_laser_,
            R_laser_,
            Q);
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  if (!is_initialized_) {
    Initialize(measurement_pack);
    cout << "EKF: " << ekf_.x_ << endl;
    return;
  }

  // dt in seconds
  float dt_s = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  ComputeProcessCovarianceMatrix(dt_s);
  ekf_.Predict();
  previous_timestamp_ = measurement_pack.timestamp_;
  Update(measurement_pack);
  
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

void FusionEKF::Initialize(const MeasurementPackage &measurement_pack) {
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    float range = measurement_pack.raw_measurements_[0];
    float angle = measurement_pack.raw_measurements_[1];
    float range_rate = measurement_pack.raw_measurements_[2];

    ekf_.x_ << range * cos(angle),
               range * sin(angle),
               range_rate * cos(angle),
               range_rate * sin(angle);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.x_ << measurement_pack.raw_measurements_[0],
               measurement_pack.raw_measurements_[1],
               0,
               0;
  }
  MaybeCorrectState();
  previous_timestamp_ = measurement_pack.timestamp_;
  is_initialized_ = true;
}

void FusionEKF::MaybeCorrectState() {
  if (fabs(ekf_.x_(0)) < kMinStateValue) {
    ekf_.x_(0) = kMinStateValue;
  }
  if (fabs(ekf_.x_(1)) < kMinStateValue) {
    ekf_.x_(1) = kMinStateValue;
  }
}

void FusionEKF::ComputeProcessCovarianceMatrix(float dt_s) {
  float dt_s2 = dt_s * dt_s;
  float dt_s3 = dt_s2 * dt_s / 2;
  float dt_s4 = dt_s3 * dt_s / 4;

  ekf_.F_ << 1, 0, dt_s, 0,
             0, 1, 0, dt_s,
             0, 0, 1, 0,
             0, 0, 0, 1;
  //ekf_.F_(0,2) = dt_s;
  //ekf_.F_(1,3) = dt_s;

  ekf_.Q_ <<
    dt_s4 * kNoiseAx, 0, dt_s3 * kNoiseAx, 0,
    0, dt_s4 * kNoiseAy, 0, dt_s3 * kNoiseAy,
    dt_s3 * kNoiseAx, 0, dt_s2 * kNoiseAx, 0,
    0, dt_s3 * kNoiseAy, 0, dt_s2 * kNoiseAy;
}

void FusionEKF::Update(const MeasurementPackage &measurement_pack) {
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
}
