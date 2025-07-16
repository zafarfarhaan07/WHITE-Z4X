function calculateCalories() {
    let age = document.getElementById('age').value;
    let weight = document.getElementById('weight').value;
    let height = document.getElementById('height').value;
    let gender = document.getElementById('gender').value;
    let activity = document.getElementById('activity').value;
    
    if (!age || !weight || !height) {
        alert("Please fill all fields.");
        return;
    }
    
    let bmr = (gender == 1)
        ? (10 * weight) + (6.25 * height) - (5 * age) + 5
        : (10 * weight) + (6.25 * height) - (5 * age) - 161;
    
    let calorieIntake = bmr * (1 + (activity * 0.2));
    document.getElementById('result').innerText = `Estimated Daily Calories: ${calorieIntake.toFixed(2)}`;
}
